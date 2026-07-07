"""
Tour Optimizer — demand arbitrage for independent artists.

Two ideas:

1. **Demand arbitrage score.** Cities where an artist's listeners-per-capita
   and supporter rate are unusually high relative to venue cost are
   "underpriced markets" — high demand, cheap to play. Big-city instinct
   (play London! play Paris!) is often wrong for a small artist: those are
   the most expensive rooms with the most competition for attention.

2. **Budget-constrained city selection.** Given a tour budget, pick the set
   of cities that maximizes expected profit using integer linear programming
   (PuLP / CBC), with a minimum-cities constraint so the tour is a tour.

Usage:
    python src/tour.py --budget 12000 --min-cities 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CITY_PATH = ROOT / "data" / "cities.csv"

# Economics assumptions (documented, adjustable): a 150–400 cap room
BASE_VENUE_COST_EUR = 900        # scaled by each city's venue_cost_index
TRAVEL_COST_PER_CITY_EUR = 350   # simplified average hop cost
TICKET_PRICE_EUR = 16
MERCH_PER_ATTENDEE_EUR = 7.5
# What fraction of a city's identified supporters + hot fans actually show up
SUPPORTER_ATTEND_RATE = 0.55
CASUAL_ATTEND_RATE = 0.02


def compute_city_economics(cities: pd.DataFrame) -> pd.DataFrame:
    df = cities.copy()

    df["expected_attendance"] = (
        df["supporters"] * SUPPORTER_ATTEND_RATE
        + (df["listeners"] - df["supporters"]) * CASUAL_ATTEND_RATE
    ).round(0)

    df["expected_revenue"] = (
        df["expected_attendance"] * (TICKET_PRICE_EUR + MERCH_PER_ATTENDEE_EUR)
    ).round(0)

    df["cost"] = (
        BASE_VENUE_COST_EUR * df["venue_cost_index"] + TRAVEL_COST_PER_CITY_EUR
    ).round(0)

    df["expected_profit"] = (df["expected_revenue"] - df["cost"]).round(0)

    # Demand arbitrage: demand intensity relative to how expensive the market is.
    demand = df["listeners_per_100k_pop"] * (1 + 3 * df["supporter_rate"])
    df["arbitrage_score"] = (demand / df["venue_cost_index"])
    df["arbitrage_score"] = (
        100 * (df["arbitrage_score"] - df["arbitrage_score"].min())
        / (df["arbitrage_score"].max() - df["arbitrage_score"].min())
    ).round(1)

    return df.sort_values("arbitrage_score", ascending=False).reset_index(drop=True)


def optimize_tour(
    econ: pd.DataFrame, budget: float = 12000, min_cities: int = 6, max_cities: int = 12
) -> dict:
    """Select cities maximizing total expected profit subject to a cost budget."""
    prob = pulp.LpProblem("tour_selection", pulp.LpMaximize)
    x = {
        row.city: pulp.LpVariable(f"play_{row.city}", cat="Binary")
        for row in econ.itertuples()
    }
    prob += pulp.lpSum(x[r.city] * r.expected_profit for r in econ.itertuples())
    prob += pulp.lpSum(x[r.city] * r.cost for r in econ.itertuples()) <= budget
    prob += pulp.lpSum(x.values()) >= min_cities
    prob += pulp.lpSum(x.values()) <= max_cities

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    chosen = [c for c, var in x.items() if var.value() == 1]
    sel = econ[econ.city.isin(chosen)].sort_values("expected_profit", ascending=False)
    return {
        "status": pulp.LpStatus[status],
        "budget_eur": budget,
        "cities": sel[
            ["city", "country", "expected_attendance", "expected_revenue",
             "cost", "expected_profit", "arbitrage_score"]
        ].to_dict(orient="records"),
        "total_cost_eur": float(sel["cost"].sum()),
        "total_expected_profit_eur": float(sel["expected_profit"].sum()),
    }


def naive_big_city_tour(econ: pd.DataFrame, budget: float, n: int) -> dict:
    """The instinctive strategy: play the biggest cities you can afford."""
    big = econ.sort_values("population_k", ascending=False)
    chosen, cost = [], 0.0
    for r in big.itertuples():
        if cost + r.cost <= budget and len(chosen) < n:
            chosen.append(r.city)
            cost += r.cost
    sel = econ[econ.city.isin(chosen)]
    return {
        "cities": chosen,
        "total_cost_eur": float(sel["cost"].sum()),
        "total_expected_profit_eur": float(sel["expected_profit"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=12000)
    parser.add_argument("--min-cities", type=int, default=6)
    parser.add_argument("--max-cities", type=int, default=12)
    args = parser.parse_args()

    cities = pd.read_csv(CITY_PATH)
    econ = compute_city_economics(cities)
    econ.to_csv(ROOT / "models" / "city_economics.csv", index=False)

    plan = optimize_tour(econ, args.budget, args.min_cities, args.max_cities)
    naive = naive_big_city_tour(econ, args.budget, len(plan["cities"]))
    plan["naive_big_city_comparison"] = naive
    uplift = plan["total_expected_profit_eur"] - naive["total_expected_profit_eur"]
    plan["profit_uplift_vs_naive_eur"] = round(uplift, 0)

    with open(ROOT / "models" / "tour_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    print(f"Optimized tour ({plan['status']}) | budget EUR {args.budget:,.0f}")
    print(f"{'City':<12} {'Attend':>7} {'Profit EUR':>11} {'Arbitrage':>10}")
    for c in plan["cities"]:
        print(f"{c['city']:<12} {c['expected_attendance']:>7.0f} "
              f"{c['expected_profit']:>11.0f} {c['arbitrage_score']:>10.1f}")
    print(f"\nTotal expected profit: EUR {plan['total_expected_profit_eur']:,.0f} "
          f"(cost EUR {plan['total_cost_eur']:,.0f})")
    print(f"Naive big-city tour profit: EUR {naive['total_expected_profit_eur']:,.0f} "
          f"({', '.join(naive['cities'][:5])}...)")
    print(f"Uplift from optimization: EUR {uplift:,.0f}")


if __name__ == "__main__":
    main()
