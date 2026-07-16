"""End-to-end pipeline orchestrator.

Chains the core modules together so both the Streamlit UI and the test suite can
run a full analysis with a single call. Emits progress via an optional callback
so the UI can drive a progress bar.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd

from core import (
    agent,
    analysis_engine,
    confidence,
    explainer,
    model_selector,
    profiler,
    report_writer,
)
from utils.logger import get_logger

log = get_logger(__name__)

ProgressFn = Optional[Callable[[float, str], None]]


def _emit(cb: ProgressFn, fraction: float, message: str) -> None:
    if cb:
        try:
            cb(fraction, message)
        except Exception:
            pass
    log.info("[%.0f%%] %s", fraction * 100, message)


def run_full_analysis(
    df: pd.DataFrame,
    question: str,
    answers: Optional[Dict[str, str]] = None,
    plan: Optional[Dict[str, Any]] = None,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Run profile -> plan -> analyses -> model -> explain -> confidence -> report.

    If ``plan`` is provided it is used directly (e.g. after the user answered
    clarifying questions); otherwise a plan is generated here.
    """
    answers = answers or {}

    _emit(progress, 0.05, "Profiling dataset...")
    profile = profiler.profile_dataset(df)

    if plan is None:
        _emit(progress, 0.20, "Planning analysis with the AI agent...")
        plan = agent.run_planning_agent(profile, question, answers)

    _emit(progress, 0.40, "Running statistical analyses...")
    results = analysis_engine.run_analyses(df, plan, profile)

    _emit(progress, 0.60, "Selecting and training the ML model...")
    model = model_selector.select_and_train(df, plan)

    _emit(progress, 0.75, "Explaining the model...")
    explanation = explainer.explain_model(model)

    _emit(progress, 0.85, "Scoring confidence for each insight...")
    confidence.annotate_results(results, df)

    _emit(progress, 0.92, "Writing the business report...")
    report = report_writer.generate_report(
        {
            "question": question,
            "plan": plan,
            "results": results,
            "model": model,
            "explanation": explanation,
        }
    )

    _emit(progress, 1.0, "Done.")
    return {
        "profile": profile,
        "plan": plan,
        "results": results,
        "model": model,
        "explanation": explanation,
        "report": report,
    }
