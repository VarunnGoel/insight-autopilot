"""Data profiling engine.

Produces a compact, JSON-serialisable profile of a DataFrame — the kind of
summary a senior analyst gathers before touching a dataset. This profile (never
the raw data) is what we send to the Gemini planning agent, which keeps token
usage and cost negligible and protects data privacy.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Heuristic keywords that hint a column is a prediction target.
_TARGET_KEYWORDS = (
    "churn",
    "target",
    "label",
    "class",
    "outcome",
    "revenue",
    "sales",
    "price",
    "profit",
    "amount",
    "score",
    "rating",
    "conversion",
    "default",
    "fraud",
    "survived",
    "response",
    "y",
)

# A categorical column with more unique values than this is treated as text/ID.
_MAX_CATEGORICAL_CARDINALITY = 50


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where obvious date-like object columns are parsed to datetime."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            name = col.lower()
            if any(tok in name for tok in ("date", "time", "day", "month", "year")):
                parsed = pd.to_datetime(out[col], errors="coerce")
                # Only accept if most values parsed successfully.
                if parsed.notna().mean() > 0.7:
                    out[col] = parsed
    return out


def _classify_column(series: pd.Series) -> str:
    """Return one of: numeric, datetime, boolean, categorical, text."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    nunique = series.nunique(dropna=True)
    if nunique <= _categorical_cap(series):
        return "categorical"
    return "text"


def _categorical_cap(series: pd.Series) -> int:
    """Cardinality cap scaled to sample size (small samples -> smaller cap)."""
    return min(_MAX_CATEGORICAL_CARDINALITY, max(2, len(series) // 2))


def column_profile(series: pd.Series) -> Dict[str, Any]:
    """Profile a single column into a JSON-safe dict."""
    col_type = _classify_column(series)
    n = int(len(series))
    n_null = int(series.isna().sum())
    profile: Dict[str, Any] = {
        "dtype": col_type,
        "null_count": n_null,
        "null_pct": round(n_null / n, 4) if n else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
    }

    if col_type == "numeric":
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean):
            profile.update(
                {
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                    "mean": float(clean.mean()),
                    "median": float(clean.median()),
                    "std": float(clean.std()) if len(clean) > 1 else 0.0,
                    "skew": float(clean.skew()) if len(clean) > 2 else 0.0,
                    "outlier_count": _iqr_outlier_count(clean),
                }
            )
    elif col_type in ("categorical", "boolean"):
        counts = series.value_counts(dropna=True).head(10)
        profile["top_values"] = {str(k): int(v) for k, v in counts.items()}
    elif col_type == "datetime":
        clean = pd.to_datetime(series, errors="coerce").dropna()
        if len(clean):
            profile["min_date"] = str(clean.min())
            profile["max_date"] = str(clean.max())

    return profile


def _iqr_outlier_count(clean: pd.Series) -> int:
    """Count values outside 1.5*IQR fences."""
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((clean < low) | (clean > high)).sum())


def top_correlations(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    """Return the top N strongest absolute Pearson correlations between numerics."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr(numeric_only=True)
    seen = set()
    pairs: List[Dict[str, Any]] = []
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if j <= i:
                continue
            value = corr.iloc[i, j]
            if pd.isna(value):
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"columns": [a, b], "correlation": round(float(value), 3)})
    pairs.sort(key=lambda d: abs(d["correlation"]), reverse=True)
    return pairs[:n]


def infer_targets(df: pd.DataFrame, profiles: Dict[str, Dict]) -> List[str]:
    """Heuristically rank columns likely to be prediction targets."""
    candidates: List[str] = []
    for col in df.columns:
        name = col.lower()
        if any(kw == name or kw in name for kw in _TARGET_KEYWORDS):
            prof = profiles.get(col, {})
            # Always include if it matches a business keyword; otherwise exclude IDs.
            kw_match = any(kw == name or kw in name for kw in _TARGET_KEYWORDS)
            if kw_match or prof.get("unique_count", 0) < max(0.95 * len(df), 1):
                candidates.append(col)
    return candidates


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Build the full dataset profile dict used across the pipeline."""
    df = _coerce_datetimes(df)
    profiles = {col: column_profile(df[col]) for col in df.columns}

    numeric_cols = [c for c, p in profiles.items() if p["dtype"] == "numeric"]
    categorical_cols = [
        c for c, p in profiles.items() if p["dtype"] in ("categorical", "boolean")
    ]
    datetime_cols = [c for c, p in profiles.items() if p["dtype"] == "datetime"]
    high_null_cols = [c for c, p in profiles.items() if p["null_pct"] > 0.2]

    profile = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": profiles,
        "correlations": top_correlations(df),
        "potential_target_columns": infer_targets(df, profiles),
        "has_datetime": bool(datetime_cols),
        "datetime_columns": datetime_cols,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "high_null_columns": high_null_cols,
    }
    log.info(
        "Profiled dataset: %s numeric, %s categorical, %s datetime columns",
        len(numeric_cols),
        len(categorical_cols),
        len(datetime_cols),
    )
    return profile
