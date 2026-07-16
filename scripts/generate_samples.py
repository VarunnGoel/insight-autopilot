"""Generate the three demo datasets using only the Python standard library.

Deterministic (seeded) so the CSVs are reproducible. Run from the project root:

    python scripts/generate_samples.py
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_datasets"
SEED = 42


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_customer_churn(n: int = 800) -> None:
    """Classification target: churn (0/1) driven by tenure, charges, support calls."""
    rng = random.Random(SEED)
    rows = []
    contracts = ["month-to-month", "one-year", "two-year"]
    for i in range(n):
        tenure = rng.randint(1, 72)
        monthly_charges = round(rng.uniform(20, 120), 2)
        contract = rng.choices(contracts, weights=[0.55, 0.25, 0.20])[0]
        support_calls = rng.randint(0, 10)
        purchase_frequency = round(_clip(rng.gauss(3, 1.5), 0.1, 8), 2)
        # Latent churn probability.
        logit = (
            2.2
            - 0.045 * tenure
            + 0.02 * monthly_charges
            + 0.35 * support_calls
            - 0.4 * purchase_frequency
            + (0.9 if contract == "month-to-month" else -0.6)
        )
        prob = 1 / (1 + math.exp(-logit / 3))
        churn = 1 if rng.random() < prob else 0
        rows.append(
            {
                "customer_id": 10000 + i,
                "tenure_months": tenure,
                "monthly_charges": monthly_charges,
                "contract_type": contract,
                "support_calls": support_calls,
                "purchase_frequency": purchase_frequency,
                "churn": churn,
            }
        )
    _write("customer_churn.csv", rows)


def generate_sales_data(days: int = 540) -> None:
    """Regression/time-series: daily revenue with trend + weekly seasonality."""
    rng = random.Random(SEED + 1)
    rows = []
    start = date(2023, 1, 1)
    regions = ["north", "south", "east", "west"]
    for d in range(days):
        current = start + timedelta(days=d)
        trend = 5000 + 6 * d  # gently rising baseline
        weekly = 800 * math.sin(2 * math.pi * (current.weekday() / 7))
        marketing_spend = round(_clip(rng.gauss(500, 150), 50, 1200), 2)
        noise = rng.gauss(0, 400)
        revenue = round(
            _clip(trend + weekly + 1.8 * marketing_spend + noise, 500, 1e9), 2
        )
        units_sold = int(_clip(revenue / rng.uniform(35, 55), 1, 1e6))
        rows.append(
            {
                "date": current.isoformat(),
                "region": rng.choice(regions),
                "marketing_spend": marketing_spend,
                "units_sold": units_sold,
                "revenue": revenue,
            }
        )
    _write("sales_data.csv", rows)


def generate_ecommerce_orders(n: int = 1000) -> None:
    """Clustering-friendly: customer order behaviour with no single target."""
    rng = random.Random(SEED + 2)
    rows = []
    categories = ["electronics", "clothing", "home", "books", "beauty"]
    # Three latent customer archetypes to make clustering meaningful.
    archetypes = [
        {
            "orders": (15, 40),
            "aov": (20, 60),
            "discount": (0.0, 0.15),
        },  # frequent bargain
        {
            "orders": (2, 8),
            "aov": (120, 400),
            "discount": (0.0, 0.05),
        },  # rare big-spender
        {
            "orders": (5, 15),
            "aov": (40, 120),
            "discount": (0.1, 0.4),
        },  # mid discount-driven
    ]
    for i in range(n):
        a = rng.choice(archetypes)
        num_orders = rng.randint(*a["orders"])
        avg_order_value = round(rng.uniform(*a["aov"]), 2)
        avg_discount = round(rng.uniform(*a["discount"]), 3)
        days_since_last = rng.randint(1, 365)
        total_spend = round(num_orders * avg_order_value * (1 - avg_discount), 2)
        rows.append(
            {
                "customer_id": 50000 + i,
                "favourite_category": rng.choice(categories),
                "num_orders": num_orders,
                "avg_order_value": avg_order_value,
                "avg_discount": avg_discount,
                "days_since_last_order": days_since_last,
                "total_spend": total_spend,
            }
        )
    _write("ecommerce_orders.csv", rows)


def _write(filename: str, rows: list) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):>5} rows -> {path}")


if __name__ == "__main__":
    generate_customer_churn()
    generate_sales_data()
    generate_ecommerce_orders()
    print("Done.")
