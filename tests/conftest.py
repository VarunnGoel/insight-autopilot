"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable when running `pytest` from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def churn_df() -> pd.DataFrame:
    path = ROOT / "data" / "sample_datasets" / "customer_churn.csv"
    return pd.read_csv(path)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    path = ROOT / "data" / "sample_datasets" / "sales_data.csv"
    return pd.read_csv(path)


@pytest.fixture
def ecommerce_df() -> pd.DataFrame:
    path = ROOT / "data" / "sample_datasets" / "ecommerce_orders.csv"
    return pd.read_csv(path)
