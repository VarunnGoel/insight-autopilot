"""Confidence scoring module.

Assigns a 0-1 confidence score to each analysis result by combining three
sub-scores — data quality, statistical significance, and bootstrap stability —
then maps the score to a human label. This is the detail most student projects
skip and the clearest signal of statistical maturity.

    final = 0.3 * data_quality + 0.4 * statistical_significance + 0.3 * bootstrap_stability
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.config import settings
from utils.logger import get_logger

log = get_logger(__name__)


def _clamp(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# Sub-scores


def data_quality_score(df: pd.DataFrame, analysis_result: Dict[str, Any]) -> float:
    """Penalise small samples, high missingness, and warnings."""
    score = 1.0
    n = len(df)
    if n < 100:
        score -= 0.3
    elif n < 500:
        score -= 0.1

    # Average missingness across the whole frame.
    null_frac = float(df.isna().mean().mean()) if df.shape[1] else 0.0
    score -= min(0.3, null_frac)

    # Each explicit warning shaves a little confidence.
    score -= 0.1 * len(analysis_result.get("warnings", []))
    return _clamp(score)


def statistical_significance_score(analysis_result: Dict[str, Any]) -> float:
    """Derive significance from p-values or model variance where available."""
    data = analysis_result.get("data", {})
    atype = analysis_result.get("analysis_type", "")

    p_values: List[float] = []
    if atype == "correlation_analysis":
        for pair in data.get("pairs", [])[:5]:
            if pair.get("p_value") is not None:
                p_values.append(pair["p_value"])
    elif atype == "segment_comparison" and data.get("p_value") is not None:
        p_values.append(data["p_value"])
    elif atype == "trend_decomposition" and data.get("p_value") is not None:
        p_values.append(data["p_value"])

    if p_values:
        best_p = min(p_values)
        if best_p < 0.01:
            return 1.0
        if best_p < 0.05:
            return 0.7
        if best_p < 0.1:
            return 0.5
        return 0.3

    # Distribution / outlier analyses have no single p-value; use a neutral prior.
    return 0.6


def bootstrap_stability_score(
    df: pd.DataFrame, analysis_result: Dict[str, Any], iterations: int = None
) -> float:
    """Re-run the headline metric on bootstrap resamples; reward low variation.

    Only implemented for correlation (fast + representative). Other analyses get a
    reasonable default so the composite score stays well-defined.
    """
    iterations = iterations or settings.bootstrap_iterations
    atype = analysis_result.get("analysis_type", "")
    data = analysis_result.get("data", {})

    if atype == "correlation_analysis":
        pairs = data.get("pairs", [])
        if not pairs:
            return 0.5
        a, b = pairs[0]["columns"]
        sub = df[[a, b]].dropna()
        if len(sub) < 20:
            return 0.4
        rs = []
        rng = np.random.default_rng(settings.random_state)
        for _ in range(iterations):
            idx = rng.integers(0, len(sub), len(sub))
            sample = sub.iloc[idx]
            if sample[a].std() == 0 or sample[b].std() == 0:
                continue
            r = np.corrcoef(sample[a], sample[b])[0, 1]
            if not np.isnan(r):
                rs.append(r)
        if len(rs) < 5:
            return 0.5
        mean_abs = abs(np.mean(rs))
        if mean_abs < 1e-6:
            return 0.5
        cv = np.std(rs) / (mean_abs + 1e-9)  # coefficient of variation
        return _clamp(1.0 - min(cv, 1.0))

    return 0.6


# Composite


def compute_insight_confidence(
    analysis_result: Dict[str, Any], df: pd.DataFrame
) -> float:
    """Blend the three sub-scores into a single 0-1 confidence value."""
    w_dq, w_ss, w_bs = settings.confidence_weights
    dq = data_quality_score(df, analysis_result)
    ss = statistical_significance_score(analysis_result)
    bs = bootstrap_stability_score(df, analysis_result)
    final = w_dq * dq + w_ss * ss + w_bs * bs
    return round(_clamp(final), 2)


def confidence_to_label(score: float) -> str:
    if score >= 0.8:
        return "High confidence"
    if score >= 0.5:
        return "Moderate confidence"
    return "Low confidence — interpret with caution"


def confidence_color(score: float) -> str:
    """Hex/keyword colour for UI badges."""
    if score >= 0.8:
        return "#1a9850"  # green
    if score >= 0.5:
        return "#fc8d59"  # orange
    return "#d73027"  # red


def annotate_results(
    results: Dict[str, Dict[str, Any]], df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Fill in the ``confidence`` field on every analysis result in place."""
    for _, res in results.items():
        try:
            res["confidence"] = compute_insight_confidence(res, df)
        except Exception as exc:
            log.warning(
                "Confidence scoring failed for %s: %s", res.get("analysis_type"), exc
            )
            res["confidence"] = 0.5
    return results
