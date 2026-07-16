"""Central configuration for Insight Autopilot.

Loads settings from environment variables (optionally via a local .env file)
and exposes a single ``settings`` object the rest of the app imports.

Everything is centralised here so there is exactly one place to change the
Claude model, tune confidence weights, or point the app at a different
sample-data directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Optional .env loading. We support python-dotenv if it is installed, but we
# also fall back to a tiny hand-rolled parser so the app never hard-fails just
# because python-dotenv is missing.

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Populate os.environ from a .env file if one exists (idempotent)."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(PROJECT_ROOT / ".env")
        return
    except Exception:
        pass

    # Fallback: minimal KEY=VALUE parser.
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Do not clobber variables already set in the real environment.
        os.environ.setdefault(key, value)


_load_dotenv()


def _get_api_key() -> str:
    """Read the LLM API key from env, or from Streamlit secrets if present."""
    key = os.environ.get("LLM_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st  # type: ignore

        return str(st.secrets.get("LLM_API_KEY", "")).strip()
    except Exception:
        return ""


@dataclass
class Settings:
    """Immutable-ish runtime settings for the whole application."""

    # --- LLM / API ---
    llm_api_key: str = field(default_factory=_get_api_key)
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "your-preferred-model-here")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.anthropic.com")
    )
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 3

    # --- Paths ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    sample_data_dir: Path = PROJECT_ROOT / "data" / "sample_datasets"
    db_path: Path = PROJECT_ROOT / "data" / "sessions.db"

    # --- ML / analysis knobs ---
    cv_splits: int = 5
    random_state: int = 42
    outlier_contamination: float = 0.05
    bootstrap_iterations: int = 50
    max_rows_for_modeling: int = 20000  # subsample very large uploads for speed

    # --- Confidence score weights (must sum to 1.0) ---
    confidence_weights: tuple = (0.3, 0.4, 0.3)  # (data_quality, stat_sig, bootstrap)

    @property
    def llm_available(self) -> bool:
        """True when we have a key and can therefore call the LLM."""
        return bool(self.llm_api_key)

    def sample_files(self) -> List[Path]:
        if not self.sample_data_dir.exists():
            return []
        return sorted(self.sample_data_dir.glob("*.csv"))


settings = Settings()
