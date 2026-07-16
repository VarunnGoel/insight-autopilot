"""SQLite session storage.

Persists each analysis run so users can revisit past analyses. Uses the stdlib
``sqlite3`` module (no SQLAlchemy needed) to keep the dependency footprint tiny.

Note: on Streamlit Cloud's free tier the filesystem is ephemeral, so this DB
resets on redeploy. The app degrades gracefully — it simply starts with an empty
history.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    dataset_name TEXT,
    row_count INTEGER,
    col_count INTEGER,
    business_question TEXT,
    problem_type TEXT,
    plan_json TEXT,
    results_json TEXT,
    report_text TEXT,
    model_performance_json TEXT
);
"""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def save_session(
    dataset_name: str,
    row_count: int,
    col_count: int,
    business_question: str,
    plan: Dict[str, Any],
    results: Dict[str, Any],
    report_text: str,
    model_performance: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> str:
    """Persist one run and return its generated session id."""
    init_db(db_path)
    session_id = uuid.uuid4().hex

    def _safe(obj: Any) -> str:
        # Drop private pipeline objects and coerce everything to JSON-safe types.
        def clean(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
            if isinstance(o, list):
                return [clean(v) for v in o]
            return o

        return json.dumps(clean(obj), default=str)

    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO sessions
               (id, created_at, dataset_name, row_count, col_count, business_question,
                problem_type, plan_json, results_json, report_text, model_performance_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id,
                datetime.now().isoformat(timespec="seconds"),
                dataset_name,
                int(row_count),
                int(col_count),
                business_question,
                plan.get("problem_type", "descriptive"),
                _safe(plan),
                _safe(results),
                report_text,
                _safe(model_performance),
            ),
        )
    log.info("Saved session %s (%s)", session_id, dataset_name)
    return session_id


def list_sessions(
    limit: int = 5, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT id, created_at, dataset_name, row_count, col_count,
                      business_question, problem_type
               FROM sessions ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_session(
    session_id: str, db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    for key in ("plan_json", "results_json", "model_performance_json"):
        try:
            data[key.replace("_json", "")] = json.loads(data.pop(key) or "{}")
        except json.JSONDecodeError:
            data[key.replace("_json", "")] = {}
    return data
