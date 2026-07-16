"""Tests for SQLite session storage (uses a temp DB, not the real one)."""

from __future__ import annotations

from core import storage


def test_save_and_load_roundtrip(tmp_path):
    db = tmp_path / "test_sessions.db"
    plan = {"problem_type": "classification", "target_column": "churn"}
    results = {
        "correlation_analysis": {"summary": "x", "_pipeline": "SHOULD_BE_DROPPED"}
    }
    sid = storage.save_session(
        dataset_name="churn.csv",
        row_count=800,
        col_count=7,
        business_question="what predicts churn?",
        plan=plan,
        results=results,
        report_text="# Report",
        model_performance={"trained": True, "_X": "DROP"},
        db_path=db,
    )
    assert sid

    loaded = storage.load_session(sid, db_path=db)
    assert loaded is not None
    assert loaded["business_question"] == "what predicts churn?"
    assert loaded["plan"]["problem_type"] == "classification"
    # Private keys must be stripped before persistence.
    assert "_pipeline" not in loaded["results"]["correlation_analysis"]
    assert "_X" not in loaded["model_performance"]


def test_list_sessions(tmp_path):
    db = tmp_path / "test_sessions.db"
    for i in range(3):
        storage.save_session(
            dataset_name=f"ds_{i}.csv",
            row_count=100,
            col_count=5,
            business_question=f"q{i}",
            plan={"problem_type": "descriptive"},
            results={},
            report_text="r",
            model_performance={},
            db_path=db,
        )
    sessions = storage.list_sessions(limit=5, db_path=db)
    assert len(sessions) == 3
