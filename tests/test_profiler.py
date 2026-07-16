"""Tests for the profiling engine."""

from __future__ import annotations

from core import profiler


def test_profile_shape_and_keys(churn_df):
    profile = profiler.profile_dataset(churn_df)
    assert profile["shape"]["rows"] == len(churn_df)
    assert profile["shape"]["cols"] == churn_df.shape[1]
    for key in ("columns", "correlations", "numeric_columns", "categorical_columns"):
        assert key in profile


def test_profile_detects_target(churn_df):
    profile = profiler.profile_dataset(churn_df)
    assert "churn" in profile["potential_target_columns"]


def test_profile_column_types(churn_df):
    profile = profiler.profile_dataset(churn_df)
    assert "monthly_charges" in profile["numeric_columns"]
    assert "contract_type" in profile["categorical_columns"]


def test_profile_detects_datetime(sales_df):
    profile = profiler.profile_dataset(sales_df)
    assert profile["has_datetime"] is True
    assert "date" in profile["datetime_columns"]


def test_correlations_are_ranked(sales_df):
    profile = profiler.profile_dataset(sales_df)
    corrs = profile["correlations"]
    if len(corrs) >= 2:
        assert abs(corrs[0]["correlation"]) >= abs(corrs[1]["correlation"])
