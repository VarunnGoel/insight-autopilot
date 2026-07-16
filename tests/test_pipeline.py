"""End-to-end pipeline tests (offline: no API key required)."""

from __future__ import annotations

from core import analysis_engine, confidence, pipeline, profiler


def test_analysis_engine_result_schema(churn_df):
    profile = profiler.profile_dataset(churn_df)
    res = analysis_engine.run_correlation_analysis(churn_df, profile["numeric_columns"])
    for key in ("analysis_type", "summary", "key_findings", "data", "warnings"):
        assert key in res


def test_confidence_scoring_range(churn_df):
    profile = profiler.profile_dataset(churn_df)
    res = analysis_engine.run_correlation_analysis(churn_df, profile["numeric_columns"])
    score = confidence.compute_insight_confidence(res, churn_df)
    assert 0.0 <= score <= 1.0
    assert confidence.confidence_to_label(score)


def test_full_pipeline_offline(churn_df):
    """The whole pipeline must run end-to-end with the offline fallbacks."""
    output = pipeline.run_full_analysis(churn_df, "What predicts customer churn?")
    assert "report" in output
    assert output["report"]["markdown"].strip()
    assert output["plan"]["problem_type"] in (
        "classification",
        "regression",
        "clustering",
        "descriptive",
    )
    # Every analysis result should have a confidence score attached.
    for res in output["results"].values():
        assert res["confidence"] is not None


def test_full_pipeline_clustering(ecommerce_df):
    output = pipeline.run_full_analysis(ecommerce_df, "What customer segments exist?")
    assert output["report"]["markdown"].strip()
