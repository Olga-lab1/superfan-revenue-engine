"""
Synthetic Fan Dataset Generator
===============================
Generates a realistic fan base for a fictional independent artist, treating
listeners as a sales pipeline: casual streamers -> engaged fans -> paying
supporters (the "superfan" conversion).

The probability that a fan becomes a paying supporter within 12 months is
driven by a documented latent model (listening depth, social engagement,
direct-channel presence, live-show exposure, geography) plus noise — so a
trained classifier has genuine signal to recover and SHAP explanations are
meaningful.

Also emits a city-level table used by the tour optimizer.

All data is synthetic. No real artists, fans, platforms' data, or people
are represented.

Usage:
    python src/generate_data.py --n 30000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fictional but plausible touring geography for a Europe-based indie artist.
# (city, country, population_k, base_listener_share, venue_cost_index)
CITIES = [
    ("Berlin", "DE", 3700, 0.075, 1.00),
    ("Hamburg", "DE", 1850, 0.038, 0.85),
    ("Leipzig", "DE", 610, 0.032, 0.55),
    ("Cologne", "DE", 1080, 0.030, 0.80),
    ("Munich", "DE", 1490, 0.028, 1.10),
    ("Amsterdam", "NL", 880, 0.050, 1.05),
    ("Rotterdam", "NL", 650, 0.020, 0.80),
    ("Utrecht", "NL", 360, 0.016, 0.70),
    ("Paris", "FR", 2140, 0.055, 1.20),
    ("Lyon", "FR", 520, 0.015, 0.75),
    ("London", "UK", 8900, 0.085, 1.35),
    ("Manchester", "UK", 550, 0.024, 0.80),
    ("Bristol", "UK", 470, 0.021, 0.75),
    ("Glasgow", "UK", 630, 0.017, 0.70),
    ("Warsaw", "PL", 1790, 0.033, 0.55),
    ("Krakow", "PL", 780, 0.022, 0.50),
    ("Prague", "CZ", 1300, 0.030, 0.60),
    ("Vienna", "AT", 1900, 0.031, 0.90),
    ("Zurich", "CH", 420, 0.014, 1.40),
    ("Copenhagen", "DK", 640, 0.026, 1.15),
    ("Stockholm", "SE", 980, 0.029, 1.10),
    ("Oslo", "NO", 700, 0.018, 1.25),
    ("Helsinki", "FI", 660, 0.019, 1.00),
    ("Riga", "LV", 615, 0.024, 0.45),
    ("Vilnius", "LT", 590, 0.020, 0.45),
    ("Tallinn", "EE", 445, 0.017, 0.50),
    ("Brussels", "BE", 1210, 0.022, 0.90),
    ("Barcelona", "ES", 1620, 0.036, 0.85),
    ("Madrid", "ES", 3300, 0.032, 0.85),
    ("Lisbon", "PT", 550, 0.021, 0.70),
    ("Milan", "IT", 1400, 0.027, 0.95),
    ("Dublin", "IE", 590, 0.019, 1.10),
]

DISCOVERY_CHANNELS = {
    # channel: (share, latent superfan boost)
    "Editorial playlist": (0.24, -0.55),   # high volume, passive listeners
    "Algorithmic radio": (0.22, -0.35),
    "Friend shared": (0.16, 0.55),
    "Live show": (0.08, 0.90),
    "Short-form video": (0.14, -0.15),
    "Music blog / press": (0.05, 0.35),
    "Bandcamp browse": (0.04, 0.65),
    "Another artist's collab": (0.07, 0.20),
}

AGE_BANDS = {
    "16-21": (0.20, -0.25),  # engaged but low purchasing power
    "22-29": (0.34, 0.20),
    "30-39": (0.26, 0.35),
    "40-54": (0.15, 0.15),
    "55+": (0.05, -0.10),
}


def generate_fans(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    city_names = [c[0] for c in CITIES]
    city_share = np.array([c[3] for c in CITIES])
    # Add an "Other / rest of world" bucket
    other_share = 1 - city_share.sum()
    city_p = np.append(city_share, other_share)
    city = rng.choice(city_names + ["Other"], size=n, p=city_p / city_p.sum())

    ch_names = list(DISCOVERY_CHANNELS)
    ch_p = np.array([DISCOVERY_CHANNELS[c][0] for c in ch_names])
    channel = rng.choice(ch_names, size=n, p=ch_p / ch_p.sum())

    age_names = list(AGE_BANDS)
    age_p = np.array([AGE_BANDS[a][0] for a in age_names])
    age_band = rng.choice(age_names, size=n, p=age_p / age_p.sum())

    # Latent affinity: how much this listener actually connects with the music
    affinity = rng.normal(0, 1, n)

    months_since_discovery = np.clip(rng.exponential(10, n), 0.5, 36).round(1)
    monthly_streams = np.clip(rng.poisson(np.exp(1.8 + 0.75 * affinity)), 0, 400)
    catalog_depth_pct = np.clip((28 + 22 * affinity + rng.normal(0, 12, n)), 1, 100).round(0)
    skip_rate = np.clip(0.38 - 0.11 * affinity + rng.normal(0, 0.08, n), 0.01, 0.95).round(3)
    playlist_adds = np.clip(rng.poisson(np.exp(0.2 + 0.6 * affinity)), 0, 40)
    saves_library = (rng.random(n) < np.clip(0.35 + 0.20 * affinity, 0.02, 0.98)).astype(int)

    follows_socials = (rng.random(n) < np.clip(0.22 + 0.18 * affinity, 0.02, 0.95)).astype(int)
    social_engagements_90d = np.where(
        follows_socials, np.clip(rng.poisson(np.exp(0.8 + 0.7 * affinity)), 0, 120), 0
    )
    on_mailing_list = (rng.random(n) < np.clip(0.08 + 0.13 * affinity, 0.01, 0.85)).astype(int)
    email_open_rate = np.where(
        on_mailing_list, np.clip(0.45 + 0.15 * affinity + rng.normal(0, 0.12, n), 0, 1), 0
    ).round(3)

    attended_show = (rng.random(n) < np.clip(0.05 + 0.10 * affinity, 0.005, 0.7)).astype(int)
    shared_track = (rng.random(n) < np.clip(0.12 + 0.15 * affinity, 0.01, 0.9)).astype(int)
    comments_dms = np.clip(rng.poisson(np.exp(-0.9 + 0.9 * affinity)), 0, 50)

    ch_boost = np.vectorize(lambda c: DISCOVERY_CHANNELS[c][1])(channel)
    age_boost = np.vectorize(lambda a: AGE_BANDS[a][1])(age_band)

    # ----- Latent superfan conversion model -----
    z = (
        -3.35
        + 0.9 * ch_boost
        + 0.8 * age_boost
        + 0.010 * monthly_streams
        + 0.016 * catalog_depth_pct
        - 1.3 * skip_rate
        + 0.07 * playlist_adds
        + 0.35 * saves_library
        + 0.55 * follows_socials
        + 0.012 * social_engagements_90d
        + 0.80 * on_mailing_list
        + 0.9 * email_open_rate
        + 1.05 * attended_show
        + 0.40 * shared_track
        + 0.05 * comments_dms
        - 0.010 * months_since_discovery  # cold fans drift away
        + rng.normal(0, 0.85, n)
    )
    p = 1 / (1 + np.exp(-z))
    became_supporter = (rng.random(n) < p).astype(int)

    # Revenue for converters: log-normal — most buy a t-shirt, a few buy everything
    revenue = np.where(
        became_supporter,
        np.clip(rng.lognormal(3.15, 0.9, n), 5, 1200),
        0.0,
    ).round(2)

    df = pd.DataFrame(
        {
            "fan_id": [f"F{200000 + i}" for i in range(n)],
            "city": city,
            "age_band": age_band,
            "discovery_channel": channel,
            "months_since_discovery": months_since_discovery,
            "monthly_streams": monthly_streams,
            "catalog_depth_pct": catalog_depth_pct,
            "skip_rate": skip_rate,
            "playlist_adds": playlist_adds,
            "saves_library": saves_library,
            "follows_socials": follows_socials,
            "social_engagements_90d": social_engagements_90d,
            "on_mailing_list": on_mailing_list,
            "email_open_rate": email_open_rate,
            "attended_show": attended_show,
            "shared_track": shared_track,
            "comments_dms": comments_dms,
            "became_supporter": became_supporter,
            "supporter_revenue_12m": revenue,
        }
    )
    return df


def build_city_table(fans: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fan data to city level and attach touring economics."""
    meta = pd.DataFrame(
        CITIES, columns=["city", "country", "population_k", "base_share", "venue_cost_index"]
    ).drop(columns=["base_share"])

    agg = (
        fans[fans.city != "Other"]
        .groupby("city")
        .agg(
            listeners=("fan_id", "size"),
            supporters=("became_supporter", "sum"),
            supporter_rate=("became_supporter", "mean"),
            avg_streams=("monthly_streams", "mean"),
            show_attenders=("attended_show", "sum"),
        )
        .reset_index()
    )
    out = agg.merge(meta, on="city")
    out["listeners_per_100k_pop"] = (out["listeners"] / out["population_k"] * 100).round(2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    fans = generate_fans(args.n, args.seed)
    fans.to_csv(outdir / "fans.csv", index=False)
    cities = build_city_table(fans)
    cities.to_csv(outdir / "cities.csv", index=False)

    conv = fans["became_supporter"].mean()
    rev = fans["supporter_revenue_12m"].sum()
    top1pct = (
        fans["supporter_revenue_12m"].nlargest(int(len(fans) * 0.01)).sum() / max(rev, 1)
    )
    print(
        f"Wrote {len(fans):,} fans | supporter conversion: {conv:.1%} | "
        f"total 12m revenue: EUR {rev:,.0f} | top 1% of fans = {top1pct:.0%} of revenue"
    )
    print(f"Wrote {len(cities)} city rows to {outdir/'cities.csv'}")


if __name__ == "__main__":
    main()
