"""Analysis engine.

Runs the specific statistical analyses named in the agent's plan. Every function
returns the SAME structured result schema so downstream consumers (confidence
scoring, charts, and the report writer) can process them uniformly:

    {
        "analysis_type": str,
        "summary": str,            # 1-2 sentence plain-English summary
        "key_findings": [str, ...],
        "data": {...},             # raw numbers for charting
        "confidence": float,       # filled in later by confidence.py
        "warnings": [str, ...],
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.formatters import strength_label, titleize
from utils.logger import get_logger

log = get_logger(__name__)


def _result(
    analysis_type: str,
    summary: str,
    findings: List[str],
    data: Dict[str, Any],
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "analysis_type": analysis_type,
        "summary": summary,
        "key_findings": findings,
        "data": data,
        "confidence": None,  # populated by confidence.compute_insight_confidence
        "warnings": warnings or [],
    }


# Correlation


def run_correlation_analysis(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    from scipy import stats

    numeric = df[[c for c in columns if pd.api.types.is_numeric_dtype(df[c])]]
    numeric = numeric.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return _result(
            "correlation_analysis",
            "Not enough numeric columns to compute correlations.",
            [],
            {"pairs": []},
            ["Need at least two numeric columns."],
        )

    corr = numeric.corr()
    pairs: List[Dict[str, Any]] = []
    cols = list(numeric.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            sub = numeric[[a, b]].dropna()
            if len(sub) < 3:
                continue
            r, p = stats.pearsonr(sub[a], sub[b])
            if np.isnan(r):
                continue
            pairs.append(
                {
                    "columns": [a, b],
                    "correlation": round(float(r), 3),
                    "p_value": round(float(p), 5),
                    "strength": strength_label(r),
                    "significant": bool(p < 0.05),
                }
            )
    pairs.sort(key=lambda d: abs(d["correlation"]), reverse=True)

    findings = []
    for pr in pairs[:5]:
        direction = "positively" if pr["correlation"] > 0 else "negatively"
        sig = "" if pr["significant"] else " (not statistically significant)"
        findings.append(
            f"{titleize(pr['columns'][0])} and {titleize(pr['columns'][1])} are "
            f"{pr['strength']}ly {direction} correlated (r={pr['correlation']}){sig}."
        )

    top = pairs[0] if pairs else None
    summary = (
        f"Strongest relationship: {titleize(top['columns'][0])} vs "
        f"{titleize(top['columns'][1])} (r={top['correlation']})."
        if top
        else "No meaningful correlations found."
    )
    return _result(
        "correlation_analysis",
        summary,
        findings,
        {"pairs": pairs, "matrix": corr.round(3).to_dict()},
    )


# Outliers


def run_outlier_detection(
    df: pd.DataFrame, columns: List[str], contamination: float = 0.05
) -> Dict[str, Any]:
    from sklearn.ensemble import IsolationForest

    numeric = df[[c for c in columns if pd.api.types.is_numeric_dtype(df[c])]]
    numeric = numeric.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[0] < 20 or numeric.shape[1] < 1:
        return _result(
            "outlier_detection",
            "Too few complete numeric rows for reliable outlier detection.",
            [],
            {"outlier_indices": []},
            ["Need >=20 complete numeric rows."],
        )

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    labels = model.fit_predict(numeric)  # -1 outlier, 1 inlier
    scores = model.score_samples(numeric)  # lower = more anomalous
    outlier_mask = labels == -1
    outlier_idx = numeric.index[outlier_mask].tolist()

    # Rank the most anomalous rows.
    order = np.argsort(scores)  # ascending -> most anomalous first
    most_anomalous = numeric.index[order][: min(5, len(order))].tolist()

    n_out = int(outlier_mask.sum())
    pct = round(n_out / len(numeric), 4)
    findings = [
        f"{n_out} rows ({pct * 100:.1f}% of complete rows) flagged as multivariate outliers.",
    ]
    # Per-column IQR outliers add univariate context.
    for col in numeric.columns:
        iqr_out = _iqr_outliers(numeric[col])
        if iqr_out:
            findings.append(
                f"{titleize(col)} has {iqr_out} univariate (IQR) outlier(s)."
            )

    return _result(
        "outlier_detection",
        f"Isolation Forest flagged {n_out} anomalous rows ({pct * 100:.1f}%).",
        findings,
        {
            "outlier_indices": outlier_idx,
            "most_anomalous_rows": most_anomalous,
            "outlier_pct": pct,
            "scores": {
                int(i): round(float(s), 4) for i, s in zip(numeric.index, scores)
            },
        },
    )


def _iqr_outliers(series: pd.Series) -> int:
    clean = series.dropna()
    if len(clean) < 4:
        return 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())


# Distributions


def run_distribution_analysis(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    from scipy import stats

    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return _result(
            "distribution_analysis",
            "No numeric columns to analyse.",
            [],
            {"columns": {}},
            ["No numeric columns present."],
        )

    per_col: Dict[str, Any] = {}
    findings: List[str] = []
    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(clean) < 8:
            continue
        skew = float(clean.skew())
        kurt = float(clean.kurt())

        # Normality test: Shapiro for small n, D'Agostino otherwise.
        try:
            if len(clean) < 5000:
                _, p_normal = stats.shapiro(
                    clean.sample(min(len(clean), 5000), random_state=42)
                )
            else:
                _, p_normal = stats.normaltest(clean)
        except Exception:
            p_normal = float("nan")

        shape = _describe_shape(skew)
        transform = _recommend_transform(skew)
        per_col[col] = {
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std()),
            "skew": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "normal_p": None if np.isnan(p_normal) else round(float(p_normal), 5),
            "shape": shape,
            "recommended_transform": transform,
            "histogram": _histogram(clean),
        }
        note = f", consider a {transform} transform" if transform else ""
        findings.append(f"{titleize(col)} is {shape}{note}.")

    return _result(
        "distribution_analysis",
        f"Analysed the distribution of {len(per_col)} numeric column(s).",
        findings,
        {"columns": per_col},
    )


def _describe_shape(skew: float) -> str:
    if skew > 1:
        return "strongly right-skewed"
    if skew > 0.5:
        return "right-skewed"
    if skew < -1:
        return "strongly left-skewed"
    if skew < -0.5:
        return "left-skewed"
    return "roughly symmetric"


def _recommend_transform(skew: float) -> Optional[str]:
    if skew > 1:
        return "log"
    if skew > 0.5:
        return "square-root"
    return None


def _histogram(clean: pd.Series, bins: int = 20) -> Dict[str, List[float]]:
    counts, edges = np.histogram(clean, bins=bins)
    return {"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]}


# Trend decomposition (lightweight, no statsmodels)


def run_trend_decomposition(
    df: pd.DataFrame, date_col: str, value_col: str
) -> Dict[str, Any]:
    """Estimate trend + seasonality using a moving average (statsmodels-free)."""
    from scipy import stats

    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna().sort_values(date_col)
    if len(work) < 12:
        return _result(
            "trend_decomposition",
            "Too few time-ordered points for trend decomposition.",
            [],
            {},
            ["Need >=12 valid time points."],
        )

    work = work.set_index(date_col)
    series = work[value_col]

    # Trend via centred moving average; window ~ 1/10 of the series.
    window = max(3, len(series) // 10)
    trend = series.rolling(window=window, center=True, min_periods=1).mean()

    # Overall direction via linear regression on time index.
    x = np.arange(len(series))
    slope, intercept, r, p, _ = stats.linregress(x, series.values)
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"

    residual = series - trend
    resid_var = float(residual.var())
    total_var = float(series.var()) or 1.0
    seasonality_strength = round(max(0.0, 1 - resid_var / total_var), 3)

    findings = [
        f"Overall trend is {direction} (slope={slope:.4f}, p={p:.4f}).",
        f"The trend is {'statistically significant' if p < 0.05 else 'not statistically significant'}.",
        f"Estimated seasonality/structure strength: {seasonality_strength}.",
    ]
    return _result(
        "trend_decomposition",
        f"{titleize(value_col)} shows a {direction} trend over time.",
        findings,
        {
            "dates": [str(d) for d in series.index],
            "values": [round(float(v), 4) for v in series.values],
            "trend": [None if pd.isna(v) else round(float(v), 4) for v in trend.values],
            "slope": round(float(slope), 6),
            "p_value": round(float(p), 5),
            "direction": direction,
            "seasonality_strength": seasonality_strength,
        },
    )


# Segment comparison (ANOVA)


def run_segment_comparison(
    df: pd.DataFrame, segment_col: str, value_col: str
) -> Dict[str, Any]:
    from scipy import stats

    work = df[[segment_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    groups = [g[value_col].values for _, g in work.groupby(segment_col) if len(g) >= 3]
    labels = [str(k) for k, g in work.groupby(segment_col) if len(g) >= 3]
    if len(groups) < 2:
        return _result(
            "segment_comparison",
            f"Not enough segments in '{segment_col}' to compare.",
            [],
            {},
            ["Need >=2 segments with >=3 rows each."],
        )

    f_stat, p_value = stats.f_oneway(*groups)
    stats_by_segment = {
        lbl: {
            "mean": float(np.mean(g)),
            "median": float(np.median(g)),
            "n": int(len(g)),
        }
        for lbl, g in zip(labels, groups)
    }
    ranked = sorted(
        stats_by_segment.items(), key=lambda kv: kv[1]["mean"], reverse=True
    )

    findings = [
        f"Highest average {titleize(value_col)}: '{ranked[0][0]}' "
        f"({ranked[0][1]['mean']:.2f}); lowest: '{ranked[-1][0]}' ({ranked[-1][1]['mean']:.2f}).",
        f"ANOVA F={f_stat:.3f}, p={p_value:.4f} — differences are "
        f"{'statistically significant' if p_value < 0.05 else 'not statistically significant'}.",
    ]
    return _result(
        "segment_comparison",
        f"Compared {titleize(value_col)} across {len(labels)} segments of {titleize(segment_col)}.",
        findings,
        {
            "segments": stats_by_segment,
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_value), 5),
            "significant": bool(p_value < 0.05),
        },
    )


# Dispatcher


def run_analyses(
    df: pd.DataFrame, plan: Dict[str, Any], profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Run every analysis named in the plan and return a dict keyed by type."""
    results: Dict[str, Any] = {}
    numeric_cols = profile.get("numeric_columns", [])
    datetime_cols = profile.get("datetime_columns", [])
    target = plan.get("target_column")

    for analysis in plan.get("analyses_to_run", []):
        try:
            if analysis == "correlation_analysis":
                results[analysis] = run_correlation_analysis(df, numeric_cols)
            elif analysis == "outlier_detection":
                results[analysis] = run_outlier_detection(df, numeric_cols)
            elif analysis == "distribution_analysis":
                results[analysis] = run_distribution_analysis(df, numeric_cols)
            elif analysis == "trend_decomposition" and datetime_cols and numeric_cols:
                val = target if target in numeric_cols else numeric_cols[0]
                results[analysis] = run_trend_decomposition(df, datetime_cols[0], val)
            elif analysis == "segment_comparison":
                seg = _pick_segment_column(profile)
                if seg and numeric_cols:
                    # Use the plan's target column for the metric compared across segments;
                    # fall back to the first non-ID numeric column.
                    val = (
                        target
                        if target in numeric_cols
                        else _pick_value_column(numeric_cols, df)
                    )
                    results[analysis] = run_segment_comparison(df, seg, val)
        except Exception as exc:  # never let one analysis crash the whole run
            log.exception("Analysis '%s' failed: %s", analysis, exc)
            results[analysis] = _result(
                analysis,
                f"Analysis failed: {exc}",
                [],
                {},
                [str(exc)],
            )
    return results


def _pick_segment_column(profile: Dict[str, Any]) -> Optional[str]:
    for col, p in profile.get("columns", {}).items():
        if (
            p.get("dtype") in ("categorical", "boolean")
            and 2 <= p.get("unique_count", 0) < 10
        ):
            return col
    return None


def _pick_value_column(numeric_cols: List[str], df: pd.DataFrame) -> str:
    """Pick the first numeric column that isn't a sequential ID."""
    for col in numeric_cols:
        name = col.lower()
        if name in ("id", "order_id", "order id") or name.endswith("_id"):
            continue
        return col
    return numeric_cols[0]
