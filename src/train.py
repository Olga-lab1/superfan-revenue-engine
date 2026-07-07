"""
Train the superfan conversion model and compute expected fan value.

Pipeline:
  1. Load data/fans.csv
  2. Stratified train/test split
  3. Baseline: logistic regression for an honest comparison
  4. Model: XGBoost (native categoricals) with early stopping
  5. Expected Fan Value = P(convert) x E[revenue | convert]
  6. Evaluate: ROC-AUC, PR-AUC, lift by decile, revenue capture curve
  7. Explain: SHAP summary + global importance
  8. Persist artifacts to models/ and charts to assets/

Usage:
    python src/train.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    REVENUE,
    TARGET,
    build_feature_frame,
)

DATA_PATH = ROOT / "data" / "fans.csv"
MODEL_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"


def lift_by_decile(y_true, y_score) -> pd.DataFrame:
    d = pd.DataFrame({"y": y_true, "score": y_score})
    d["decile"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False) + 1
    base = d["y"].mean()
    t = (
        d.groupby("decile")
        .agg(fans=("y", "size"), converts=("y", "sum"), conv_rate=("y", "mean"))
        .sort_index(ascending=False)
    )
    t["lift"] = t["conv_rate"] / base
    t["cum_converts_pct"] = t["converts"].cumsum() / d["y"].sum()
    return t.round(3)


def revenue_capture_curve(revenue: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """If you only nurture the top X% of fans by score, what share of revenue do you reach?"""
    order = np.argsort(-score)
    rev_sorted = revenue[order]
    cum = np.cumsum(rev_sorted) / max(rev_sorted.sum(), 1e-9)
    pct = np.arange(1, len(cum) + 1) / len(cum)
    idx = [int(len(cum) * q) - 1 for q in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)]
    return pd.DataFrame(
        {"top_pct_of_fans": ["1%", "2%", "5%", "10%", "20%", "50%"],
         "revenue_captured": [round(float(cum[i]), 3) for i in idx]}
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=42
    )

    X_train, X_test = build_feature_frame(train_df), build_feature_frame(test_df)
    y_train, y_test = train_df[TARGET].to_numpy(), test_df[TARGET].to_numpy()

    # ---------- Baseline ----------
    baseline = Pipeline(
        [
            ("prep", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ])),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    baseline.fit(X_train, y_train)
    base_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

    # ---------- XGBoost conversion model ----------
    cut = int(len(X_train) * 0.85)
    model = xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=5, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.8, reg_lambda=2.0,
        enable_categorical=True, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=60, random_state=42,
    )
    model.fit(
        X_train.iloc[:cut], y_train[:cut],
        eval_set=[(X_train.iloc[cut:], y_train[cut:])], verbose=False,
    )
    proba = model.predict_proba(X_test)[:, 1]

    # ---------- Conditional revenue model (converters only) ----------
    conv_train = train_df[train_df[TARGET] == 1]
    rev_model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        enable_categorical=True, tree_method="hist", random_state=42,
    )
    rev_model.fit(build_feature_frame(conv_train), np.log1p(conv_train[REVENUE]))
    expected_rev_if_convert = np.expm1(rev_model.predict(X_test))
    expected_fan_value = proba * expected_rev_if_convert

    # ---------- Metrics ----------
    lift = lift_by_decile(y_test, proba)
    capture = revenue_capture_curve(test_df[REVENUE].to_numpy(), expected_fan_value)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "baseline_logreg_roc_auc": round(float(base_auc), 4),
        "base_conversion_rate": round(float(y_test.mean()), 4),
        "top_decile_lift": float(lift.iloc[0]["lift"]),
        "top_2_deciles_capture_pct": round(float(lift.iloc[:2]["converts"].sum() / y_test.sum()), 4),
        "revenue_capture_top_10pct": float(capture.loc[capture.top_pct_of_fans == "10%", "revenue_captured"].iloc[0]),
        "best_iteration": int(model.best_iteration),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # ---------- SHAP ----------
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(n=min(2500, len(X_test)), random_state=42)
    sv = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    global_importance = (
        pd.Series(np.abs(sv).mean(axis=0), index=sample.columns)
        .sort_values(ascending=False).round(4).to_dict()
    )

    # ---------- Charts ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(d) for d in range(10, 0, -1)], lift["conv_rate"] * 100, color="#7c3aed")
    ax.axhline(y_test.mean() * 100, color="#dc2626", ls="--",
               label=f"Baseline {y_test.mean():.0%}")
    ax.set_xlabel("Fan score decile (10 = highest)")
    ax.set_ylabel("Supporter conversion rate (%)")
    ax.set_title("Superfan conversion by model score decile (holdout)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "lift_by_decile.png", dpi=160)
    plt.close(fig)

    rev = test_df[REVENUE].to_numpy()
    order = np.argsort(-expected_fan_value)
    cum = np.cumsum(rev[order]) / max(rev.sum(), 1e-9)
    pct = np.arange(1, len(cum) + 1) / len(cum)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pct * 100, cum * 100, color="#7c3aed", lw=2, label="Ranked by Expected Fan Value")
    ax.plot([0, 100], [0, 100], color="#9ca3af", ls="--", label="Random outreach")
    ax.set_xlabel("% of fan base contacted (best first)")
    ax.set_ylabel("% of supporter revenue reached")
    ax.set_title("Revenue capture curve — the superfan concentration effect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "revenue_capture.png", dpi=160)
    plt.close(fig)

    # ---------- Persist ----------
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / "superfan_model.json")
    rev_model.save_model(MODEL_DIR / "revenue_model.json")
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "global_shap_importance": global_importance,
                "revenue_capture": capture.to_dict(orient="records"),
                "categorical_levels": {
                    c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES
                },
            },
            f, indent=2,
        )
    lift.to_csv(MODEL_DIR / "lift_table.csv")

    print(json.dumps(metrics, indent=2))
    print("\nRevenue capture (ranked by Expected Fan Value):")
    print(capture.to_string(index=False))


if __name__ == "__main__":
    main()
