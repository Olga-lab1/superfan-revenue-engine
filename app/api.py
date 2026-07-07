"""
Superfan Revenue Engine API
===========================
FastAPI service exposing the two halves of the engine:

  POST /score-fan    -> superfan conversion probability + expected fan value
                        + SHAP drivers + recommended action
  POST /tour-plan    -> budget-constrained optimal tour (integer programming)
  GET  /health       -> liveness + model metadata

Run locally:
    uvicorn app.api:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import CATEGORICAL_FEATURES, build_feature_frame  # noqa: E402
from src.tour import compute_city_economics, optimize_tour  # noqa: E402

MODEL_PATH = ROOT / "models" / "superfan_model.json"
REV_MODEL_PATH = ROOT / "models" / "revenue_model.json"
META_PATH = ROOT / "models" / "metadata.json"
CITY_PATH = ROOT / "data" / "cities.csv"

_model: xgb.XGBClassifier | None = None
_rev_model: xgb.XGBRegressor | None = None
_explainer: shap.TreeExplainer | None = None
_meta: dict = {}
_city_econ: pd.DataFrame | None = None


def _load() -> None:
    global _model, _rev_model, _explainer, _meta, _city_econ
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Models not found. Run `python src/train.py` first.")
        _model = xgb.XGBClassifier(enable_categorical=True)
        _model.load_model(MODEL_PATH)
        _rev_model = xgb.XGBRegressor(enable_categorical=True)
        _rev_model.load_model(REV_MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)
        _meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        _city_econ = compute_city_economics(pd.read_csv(CITY_PATH))


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(
    title="Superfan Revenue Engine",
    description=(
        "RevOps for independent artists: score every fan for superfan conversion, "
        "compute expected fan value, and plan tours with demand-arbitrage optimization."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class AgeBand(str, Enum):
    a1 = "16-21"
    a2 = "22-29"
    a3 = "30-39"
    a4 = "40-54"
    a5 = "55+"


class Channel(str, Enum):
    editorial = "Editorial playlist"
    algo = "Algorithmic radio"
    friend = "Friend shared"
    live = "Live show"
    video = "Short-form video"
    blog = "Music blog / press"
    bandcamp = "Bandcamp browse"
    collab = "Another artist's collab"


class Fan(BaseModel):
    """One listener as seen across streaming, social, and direct channels."""

    fan_id: str = Field(default="unknown", examples=["F123456"])
    city: str = Field(examples=["Leipzig"])
    age_band: AgeBand
    discovery_channel: Channel
    months_since_discovery: float = Field(ge=0, le=120, examples=[6])
    monthly_streams: int = Field(ge=0, le=2000, examples=[45])
    catalog_depth_pct: float = Field(ge=0, le=100, examples=[62])
    skip_rate: float = Field(ge=0, le=1, examples=[0.12])
    playlist_adds: int = Field(ge=0, le=200, examples=[4])
    saves_library: int = Field(ge=0, le=1, examples=[1])
    follows_socials: int = Field(ge=0, le=1, examples=[1])
    social_engagements_90d: int = Field(ge=0, le=1000, examples=[12])
    on_mailing_list: int = Field(ge=0, le=1, examples=[1])
    email_open_rate: float = Field(ge=0, le=1, examples=[0.7])
    attended_show: int = Field(ge=0, le=1, examples=[0])
    shared_track: int = Field(ge=0, le=1, examples=[1])
    comments_dms: int = Field(ge=0, le=500, examples=[3])


class Driver(BaseModel):
    feature: str
    value: str
    shap_impact: float
    direction: str


class FanScore(BaseModel):
    fan_id: str
    superfan_probability: float
    expected_fan_value_eur: float
    tier: str
    recommended_action: str
    top_positive_drivers: list[Driver]
    top_negative_drivers: list[Driver]


TIERS = [
    (0.40, "Superfan-ready", "Personal invite: presale access, signed edition, direct thank-you."),
    (0.20, "Warming", "Move to owned channels: mailing-list invite with exclusive track."),
    (0.08, "Casual+", "Retarget with live session video; nudge toward a library save."),
    (0.00, "Passive", "Keep in broad reach campaigns only; no direct spend."),
]


def _tier(p: float) -> tuple[str, str]:
    for threshold, name, action in TIERS:
        if p >= threshold:
            return name, action
    return TIERS[-1][1], TIERS[-1][2]


def _drivers(shap_row: np.ndarray, x_row: pd.Series, k: int = 3):
    order = np.argsort(shap_row)
    neg = [i for i in order[:k] if shap_row[i] < 0]
    pos = [i for i in order[::-1][:k] if shap_row[i] > 0]

    def mk(i, direction):
        return Driver(
            feature=x_row.index[i], value=str(x_row.iloc[i]),
            shap_impact=round(float(shap_row[i]), 4), direction=direction,
        )

    return ([mk(i, "increases score") for i in pos],
            [mk(i, "decreases score") for i in neg])


@app.get("/health")
def health() -> dict:
    _load()
    return {"status": "ok", "models": ["superfan_xgb", "revenue_xgb"],
            "metrics": _meta.get("metrics", {})}


@app.post("/score-fan", response_model=FanScore)
def score_fan(fan: Fan) -> FanScore:
    _load()
    df = pd.DataFrame([fan.model_dump()])
    X = build_feature_frame(df)
    levels = _meta.get("categorical_levels", {})
    for col in CATEGORICAL_FEATURES:
        if col in levels:
            X[col] = pd.Categorical(X[col], categories=levels[col])

    p = float(_model.predict_proba(X)[:, 1][0])
    rev_if_convert = float(np.expm1(_rev_model.predict(X))[0])
    efv = round(p * max(rev_if_convert, 0.0), 2)
    tier, action = _tier(p)
    pos, neg = _drivers(_explainer.shap_values(X)[0], X.iloc[0])

    return FanScore(
        fan_id=fan.fan_id,
        superfan_probability=round(p, 4),
        expected_fan_value_eur=efv,
        tier=tier,
        recommended_action=action,
        top_positive_drivers=pos,
        top_negative_drivers=neg,
    )


class TourRequest(BaseModel):
    budget_eur: float = Field(gt=0, le=1_000_000, examples=[12000])
    min_cities: int = Field(ge=1, le=30, default=6)
    max_cities: int = Field(ge=1, le=30, default=12)


@app.post("/tour-plan")
def tour_plan(req: TourRequest) -> dict:
    _load()
    if req.min_cities > req.max_cities:
        raise HTTPException(422, "min_cities cannot exceed max_cities.")
    plan = optimize_tour(_city_econ, req.budget_eur, req.min_cities, req.max_cities)
    if plan["status"] != "Optimal":
        raise HTTPException(
            422, f"No feasible tour for these constraints (solver: {plan['status']}). "
                 "Try a higher budget or fewer minimum cities.")
    return plan
