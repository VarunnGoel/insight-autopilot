"""Tests for automatic model selection and training."""

from __future__ import annotations

from core import agent, model_selector, profiler


def test_classification_training(churn_df):
    plan = {
        "problem_type": "classification",
        "target_column": "churn",
        "feature_columns": [
            c for c in churn_df.columns if c not in ("churn", "customer_id")
        ],
        "analyses_to_run": [],
    }
    result = model_selector.select_and_train(churn_df, plan)
    assert result["problem_type"] == "classification"
    assert result["trained"] is True
    assert -0.1 <= result["cv_mean"] <= 1.0
    assert result["feature_importances"]  # non-empty


def test_regression_training(sales_df):
    plan = {
        "problem_type": "regression",
        "target_column": "revenue",
        "feature_columns": ["marketing_spend", "units_sold", "region"],
        "analyses_to_run": [],
    }
    result = model_selector.select_and_train(sales_df, plan)
    assert result["problem_type"] == "regression"
    assert result["trained"] is True
    assert result["metric_name"] == "R²"


def test_clustering_training(ecommerce_df):
    plan = {
        "problem_type": "clustering",
        "target_column": None,
        "feature_columns": [
            "num_orders",
            "avg_order_value",
            "avg_discount",
            "days_since_last_order",
            "total_spend",
        ],
        "analyses_to_run": [],
    }
    result = model_selector.select_and_train(ecommerce_df, plan)
    assert result["problem_type"] == "clustering"
    assert result["trained"] is True
    assert 2 <= result["best_k"] <= 8


def test_heuristic_planner_offline(churn_df):
    """With no API key, the heuristic planner should still produce a valid plan."""
    profile = profiler.profile_dataset(churn_df)
    plan = agent._heuristic_plan(profile, "what predicts churn?")
    assert plan["problem_type"] in ("classification", "regression", "descriptive")
    assert plan["planner"] == "heuristic"
    assert isinstance(plan["analyses_to_run"], list)
