"""Data ingestion layer.

Loads data from CSV, Excel, or a SQL connection into a pandas DataFrame,
normalises column names, and reports basic load metadata. Accepts either a
filesystem path or a file-like object (as Streamlit's uploader provides).
"""

from __future__ import annotations

import io
import re
from typing import Dict, Optional, Union

import pandas as pd

from utils.logger import get_logger
from utils.validators import validate_dataframe

log = get_logger(__name__)

# A "source" is either a path string or an in-memory buffer (Streamlit upload).
Source = Union[str, io.BytesIO, "io.BufferedReader"]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, trim, and replace non-alphanumerics with '_'."""

    def clean(name: object) -> str:
        text = str(name).strip().lower()
        text = re.sub(r"[^\w]+", "_", text)  # spaces, punctuation -> underscore
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "column"

    new_cols = {}
    seen: Dict[str, int] = {}
    for col in df.columns:
        base = clean(col)
        if base in seen:
            seen[base] += 1
            new_cols[col] = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            new_cols[col] = base
    return df.rename(columns=new_cols)


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read CSV bytes, trying encodings and auto-detecting the delimiter."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            # sep=None + engine='python' triggers delimiter sniffing.
            return pd.read_csv(
                io.BytesIO(data), sep=None, engine="python", encoding=enc
            )
        except Exception as exc:  # UnicodeDecodeError, ParserError, etc.
            last_err = exc
            continue
    raise ValueError(f"Could not parse CSV with any known encoding: {last_err}")


def load_csv(source: Source) -> pd.DataFrame:
    """Load a CSV from a path or file-like object with encoding detection."""
    if isinstance(source, str):
        with open(source, "rb") as fh:
            data = fh.read()
    else:
        data = source.read()
    df = _read_csv_bytes(data)
    return normalise_columns(df)


def load_excel(source: Source, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load an Excel file. If no sheet is given, pick the one with most rows."""
    xls = pd.ExcelFile(source)
    if sheet_name is None:
        # Choose the largest sheet by row count.
        best_sheet, best_rows = xls.sheet_names[0], -1
        for name in xls.sheet_names:

            try:
                full = xls.parse(name)
            except Exception:
                continue
            if len(full) > best_rows:
                best_sheet, best_rows = name, len(full)
        sheet_name = best_sheet
    df = xls.parse(sheet_name)
    return normalise_columns(df)


def load_sql(connection_string: str, query: str) -> pd.DataFrame:
    """Run a SQL query and return the result as a DataFrame.

    Uses SQLAlchemy so the same code works for SQLite, PostgreSQL, and MySQL.
    Example connection string: ``sqlite:///data/sessions.db``.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover - only if dep missing
        raise ImportError(
            "SQL loading requires SQLAlchemy. Install with `pip install sqlalchemy`."
        ) from exc

    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return normalise_columns(df)


def load_any(
    source: Source,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> Dict:
    """Dispatch to the right loader based on file extension and validate.

    Returns: {"df": DataFrame, "validation": {...}, "metadata": {...}}.
    """
    name = (filename or (source if isinstance(source, str) else "") or "").lower()

    if name.endswith((".xlsx", ".xls")):
        df = load_excel(source, sheet_name=sheet_name)
        kind = "excel"
    else:
        df = load_csv(source)
        kind = "csv"

    validation = validate_dataframe(df)
    metadata = {
        "source_kind": kind,
        "filename": filename or (source if isinstance(source, str) else "uploaded"),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    log.info(
        "Loaded %s: %s rows x %s cols",
        metadata["filename"],
        metadata["rows"],
        metadata["cols"],
    )
    return {"df": df, "validation": validation, "metadata": metadata}
