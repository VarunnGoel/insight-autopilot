"""Input validation helpers used by ingestion and the UI."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

MIN_ROWS = 10
MIN_COLS = 2


def validate_dataframe(df: pd.DataFrame) -> Dict:
    """Run basic sanity checks on a loaded DataFrame.

    Returns a dict: {"valid": bool, "warnings": [...], "errors": [...],
    "row_count": int, "col_count": int}.
    """
    warnings: List[str] = []
    errors: List[str] = []

    row_count = int(df.shape[0])
    col_count = int(df.shape[1])

    if row_count < MIN_ROWS:
        errors.append(
            f"Dataset has only {row_count} rows; at least {MIN_ROWS} are needed "
            "for meaningful analysis."
        )
    if col_count < MIN_COLS:
        errors.append(
            f"Dataset has only {col_count} column(s); at least {MIN_COLS} are needed."
        )

    # Fully-empty columns.
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        warnings.append(f"Columns entirely empty (will be ignored): {empty_cols}")

    # Columns with very high missingness.
    high_null = [
        str(c) for c in df.columns if df[c].isna().mean() > 0.5 and c not in empty_cols
    ]
    if high_null:
        warnings.append(f"Columns with >50% missing values: {high_null}")

    # Duplicate rows.
    dup = int(df.duplicated().sum())
    if dup:
        warnings.append(f"{dup} fully-duplicated row(s) detected.")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "row_count": row_count,
        "col_count": col_count,
    }


def is_valid_business_question(question: str) -> bool:
    """A question is usable if it is non-trivial free text."""
    return bool(question) and len(question.strip()) >= 5
