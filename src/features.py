"""
Feature engineering for the superfan conversion model.

Shared between training (src/train.py) and serving (app/api.py) so the exact
same transformations apply at fit time and score time — no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["city", "age_band", "discovery_channel"]

NUMERIC_FEATURES = [
    "months_since_discovery",
    "monthly_streams",
    "catalog_depth_pct",
    "skip_rate",
    "playlist_adds",
    "saves_library",
    "follows_socials",
    "social_engagements_90d",
    "on_mailing_list",
    "email_open_rate",
    "attended_show",
    "shared_track",
    "comments_dms",
]

ENGINEERED_FEATURES = [
    "listening_intensity",
    "direct_channel_score",
    "advocacy_score",
    "engagement_velocity",
]

TARGET = "became_supporter"
REVENUE = "supporter_revenue_12m"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. Works on both training frames and 1-row scoring frames."""
    out = df.copy()
    # Depth x volume x completion: how seriously do they actually listen?
    out["listening_intensity"] = (
        out["monthly_streams"] * (out["catalog_depth_pct"] / 100) * (1 - out["skip_rate"])
    ).round(2)
    # Owned channels beat rented reach: mailing list is worth more than a follow
    out["direct_channel_score"] = (
        2.0 * out["on_mailing_list"] * (0.5 + out["email_open_rate"])
        + 1.0 * out["follows_socials"]
    ).round(3)
    # Fans who spread the music are worth more than fans who consume it
    out["advocacy_score"] = (
        1.5 * out["shared_track"] + 0.5 * np.log1p(out["comments_dms"]) + 1.0 * out["playlist_adds"].clip(upper=10) / 10
    ).round(3)
    # Engagement normalized by tenure: a new fan with 5 engagements > old fan with 5
    out["engagement_velocity"] = (
        out["social_engagements_90d"] / out["months_since_discovery"].clip(lower=1)
    ).round(3)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_engineered_features(df)
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ENGINEERED_FEATURES
    X = df[features].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def feature_names() -> list[str]:
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES + ENGINEERED_FEATURES
