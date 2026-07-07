"""Test suite: data generation, features, model quality gates, tour optimizer, API contract."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate_data import generate_fans, build_city_table
from src.features import build_feature_frame, feature_names, TARGET
from src.tour import compute_city_economics, optimize_tour, naive_big_city_tour


# ---------- Data generation ----------

def test_generator_reproducible():
    a = generate_fans(400, seed=7)
    b = generate_fans(400, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_schema_and_ranges():
    df = generate_fans(2000, seed=1)
    assert df["fan_id"].is_unique
    assert df[TARGET].isin([0, 1]).all()
    assert (df["skip_rate"].between(0, 1)).all()
    assert 0.05 < df[TARGET].mean() < 0.30
    # non-converters have zero revenue; converters positive
    assert (df.loc[df[TARGET] == 0, "supporter_revenue_12m"] == 0).all()
    assert (df.loc[df[TARGET] == 1, "supporter_revenue_12m"] > 0).all()


def test_known_signals_exist():
    df = generate_fans(8000, seed=3)
    ml = df[df.on_mailing_list == 1][TARGET].mean()
    no_ml = df[df.on_mailing_list == 0][TARGET].mean()
    assert ml > no_ml
    live = df[df.discovery_channel == "Live show"][TARGET].mean()
    editorial = df[df.discovery_channel == "Editorial playlist"][TARGET].mean()
    assert live > editorial


def test_revenue_concentration():
    """The core thesis: a small share of fans generates a large share of revenue."""
    df = generate_fans(10000, seed=5)
    rev = df["supporter_revenue_12m"].to_numpy()
    top5 = np.sort(rev)[::-1][: int(len(rev) * 0.05)].sum()
    assert top5 / rev.sum() > 0.5


# ---------- Features ----------

def test_feature_frame():
    df = generate_fans(300, seed=2)
    X = build_feature_frame(df)
    assert list(X.columns) == feature_names()
    assert not X.isna().any().any()
    assert str(X["city"].dtype) == "category"


def test_feature_frame_single_row():
    X = build_feature_frame(generate_fans(1, seed=9))
    assert len(X) == 1
    assert np.isfinite(X.select_dtypes("number").to_numpy()).all()


# ---------- Tour optimizer ----------

@pytest.fixture(scope="module")
def econ():
    fans = generate_fans(20000, seed=42)
    return compute_city_economics(build_city_table(fans))


def test_optimizer_respects_budget(econ):
    plan = optimize_tour(econ, budget=8000, min_cities=4, max_cities=10)
    assert plan["status"] == "Optimal"
    assert plan["total_cost_eur"] <= 8000
    assert 4 <= len(plan["cities"]) <= 10


def test_optimizer_beats_naive(econ):
    plan = optimize_tour(econ, budget=12000, min_cities=6, max_cities=12)
    naive = naive_big_city_tour(econ, budget=12000, n=len(plan["cities"]))
    assert plan["total_expected_profit_eur"] >= naive["total_expected_profit_eur"]


def test_optimizer_infeasible_constraints(econ):
    plan = optimize_tour(econ, budget=500, min_cities=10, max_cities=12)
    assert plan["status"] != "Optimal"


# ---------- Trained model quality gate ----------

MODEL_PATH = ROOT / "models" / "superfan_model.json"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_model_beats_random():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(MODEL_PATH)
    df = generate_fans(4000, seed=123)  # unseen seed
    proba = model.predict_proba(build_feature_frame(df))[:, 1]
    assert roc_auc_score(df[TARGET], proba) > 0.72


# ---------- API contract ----------

HOT_FAN = {
    "fan_id": "F_TEST",
    "city": "Leipzig",
    "age_band": "30-39",
    "discovery_channel": "Live show",
    "months_since_discovery": 5,
    "monthly_streams": 60,
    "catalog_depth_pct": 70,
    "skip_rate": 0.08,
    "playlist_adds": 6,
    "saves_library": 1,
    "follows_socials": 1,
    "social_engagements_90d": 25,
    "on_mailing_list": 1,
    "email_open_rate": 0.8,
    "attended_show": 1,
    "shared_track": 1,
    "comments_dms": 5,
}


@pytest.fixture(scope="module")
def client():
    if not MODEL_PATH.exists():
        pytest.skip("model not trained yet")
    from app.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_hot_fan(client):
    r = client.post("/score-fan", json=HOT_FAN)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["superfan_probability"] <= 1
    assert b["expected_fan_value_eur"] >= 0
    assert b["tier"] in ("Superfan-ready", "Warming")
    assert len(b["top_positive_drivers"]) >= 1


def test_passive_fan_scores_lower(client):
    passive = dict(HOT_FAN)
    passive.update(
        discovery_channel="Editorial playlist", monthly_streams=3,
        catalog_depth_pct=5, skip_rate=0.7, playlist_adds=0, saves_library=0,
        follows_socials=0, social_engagements_90d=0, on_mailing_list=0,
        email_open_rate=0, attended_show=0, shared_track=0, comments_dms=0,
    )
    hot = client.post("/score-fan", json=HOT_FAN).json()
    cold = client.post("/score-fan", json=passive).json()
    assert cold["superfan_probability"] < hot["superfan_probability"]
    assert cold["expected_fan_value_eur"] < hot["expected_fan_value_eur"]


def test_score_rejects_bad_enum(client):
    bad = dict(HOT_FAN, discovery_channel="Telepathy")
    assert client.post("/score-fan", json=bad).status_code == 422


def test_tour_plan_endpoint(client):
    r = client.post("/tour-plan", json={"budget_eur": 12000, "min_cities": 6, "max_cities": 12})
    assert r.status_code == 200
    b = r.json()
    assert b["total_cost_eur"] <= 12000
    assert len(b["cities"]) >= 6


def test_tour_plan_invalid_constraints(client):
    r = client.post("/tour-plan", json={"budget_eur": 12000, "min_cities": 10, "max_cities": 4})
    assert r.status_code == 422
