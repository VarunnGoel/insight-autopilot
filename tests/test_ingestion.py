"""Tests for the ingestion layer."""

from __future__ import annotations

import io

import pandas as pd

from core import ingestion


def test_normalise_columns_dedupes_and_cleans():
    df = pd.DataFrame({"First Name": [1], "first name": [2], "Weird!!Col": [3]})
    out = ingestion.normalise_columns(df)
    cols = list(out.columns)
    assert "first_name" in cols
    # Duplicate normalised name gets a numeric suffix.
    assert any(c.startswith("first_name_") for c in cols)
    assert "weird_col" in cols


def test_load_csv_from_buffer():
    raw = b"A,B\n1,2\n3,4\n"
    df = ingestion.load_csv(io.BytesIO(raw))
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_load_csv_semicolon_delimiter():
    raw = b"a;b;c\n1;2;3\n4;5;6\n"
    df = ingestion.load_csv(io.BytesIO(raw))
    assert df.shape == (2, 3)


def test_load_any_reports_validation(churn_df, tmp_path):
    path = tmp_path / "churn.csv"
    churn_df.to_csv(path, index=False)
    loaded = ingestion.load_any(str(path), filename="churn.csv")
    assert loaded["validation"]["valid"] is True
    assert loaded["metadata"]["rows"] == len(churn_df)
    assert loaded["df"].shape[1] == churn_df.shape[1]
