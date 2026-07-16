"""AI planning agent.

Given a dataset profile and a plain-English business question, this module
produces a structured analysis plan (JSON) that drives the rest of the
pipeline. Claude does the reasoning when a key is available; otherwise a
deterministic heuristic planner keeps the whole app working offline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


from utils.llm import LLMError, generate_json, llm_is_available
from utils.logger import get_logger

log = get_logger(__name__)

AVAILABLE_ANALYSES = [
    "correlation_analysis",
    "outlier_detection",
    "distribution_analysis",
    "trend_decomposition",
    "segment_comparison",
]

VALID_PROBLEM_TYPES = ("regression", "classification", "clustering", "descriptive")

PLANNING_SYSTEM_PROMPT = """\
You are a senior data analyst. You receive (1) a JSON profile of a dataset and
(2) a business question. Output a structured analysis plan as a single JSON
object and nothing else.

CRITICAL: You MUST pick analysis names EXACTLY from this list — no variations:
- "correlation_analysis"
- "outlier_detection"
- "distribution_analysis"
- "trend_decomposition"
- "segment_comparison"

Schema:
{
  "problem_type": "regression" | "classification" | "clustering" | "descriptive",
  "target_column": "<column name>" | null,
  "feature_columns": ["col1", "col2", ...],
  "analyses_to_run": ["exact_name_1", "exact_name_2", ...],
  "clarifying_notes": "caveats about data quality or the question",
  "reasoning": "1-2 sentences specific to THIS dataset"
}

Rules for picking analysis names:
- Only use "trend_decomposition" if the profile has datetime columns.
- Only use "segment_comparison" if at least one categorical column has < 10 unique values.
- "correlation_analysis" and "distribution_analysis" work on any dataset with numeric columns.

Rules for problem_type:
- If the question asks about predicting a numeric value (revenue, price, score, satisfaction),
  use "regression" with that column as target.
- If the question asks about predicting a category (churn, yes/no, return, segment),
  use "classification" with that column as target.
- If the question asks to find groups or segments, use "clustering".
- For comparative business questions ("best region", "which product", "top segment") where a
  numeric metric like revenue, satisfaction, or profit exists: use "regression" with that
  metric as target_column. The comparison becomes a feature in the model.
- Use "descriptive" only when there is genuinely no column that could serve as a prediction target.
- feature_columns must be real column names from the profile.
- reasoning must reference something specific about THIS dataset.
"""

CLARIFY_SYSTEM_PROMPT = """\
You are a senior data analyst preparing to analyse a dataset. Based on the
dataset profile and the user's question, produce 2-3 short clarifying questions
that would materially change how you analyse the data. Output ONLY a JSON object:
{"questions": [{"question": "...", "options": ["opt1", "opt2", "opt3"]}]}
Keep each question answerable by picking one option.
"""


# Clarifying questions


def generate_clarifying_questions(profile: Dict, question: str) -> List[Dict[str, Any]]:
    """Return a small list of clarifying questions (with options) for the UI."""
    if llm_is_available():
        try:
            prompt = (
                f"Dataset profile:\n{json.dumps(_slim_profile(profile), indent=2)}\n\n"
                f"Business question: {question}\n\nGenerate the clarifying questions."
            )
            result = generate_json(prompt, system=CLARIFY_SYSTEM_PROMPT, max_tokens=512)
            questions = result.get("questions", [])
            if isinstance(questions, list) and questions:
                return questions[:3]
        except LLMError as exc:
            log.warning(
                "Clarifying-question generation failed, using fallback: %s", exc
            )
    return _fallback_clarifying_questions(profile)


def _fallback_clarifying_questions(profile: Dict) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    targets = profile.get("potential_target_columns") or profile.get(
        "numeric_columns", []
    )
    if targets:
        questions.append(
            {
                "question": f"Is a higher value of '{targets[0]}' good or bad for the business?",
                "options": [
                    "Higher is better",
                    "Higher is worse",
                    "Not sure / neutral",
                ],
            }
        )
    if profile.get("has_datetime"):
        questions.append(
            {
                "question": "Is this data tracked over time for the same entities, or a single snapshot?",
                "options": [
                    "Over time (longitudinal)",
                    "Single snapshot (cross-sectional)",
                ],
            }
        )
    questions.append(
        {
            "question": "Should ID-like columns be excluded from modeling?",
            "options": ["Yes, exclude IDs", "No, keep all columns"],
        }
    )
    return questions[:3]


# Planning


def run_planning_agent(
    profile: Dict, question: str, answers: Dict[str, str]
) -> Dict[str, Any]:
    """Produce a validated analysis plan dict."""
    if llm_is_available():
        try:
            plan = _plan_with_claude(profile, question, answers)
            return _validate_and_repair_plan(plan, profile)
        except LLMError as exc:
            log.warning("Claude planning failed, using heuristic planner: %s", exc)
    return _heuristic_plan(profile, question)


def _plan_with_claude(profile: Dict, question: str, answers: Dict[str, str]) -> Dict:
    prompt = (
        f"Dataset Profile:\n{json.dumps(_slim_profile(profile), indent=2)}\n\n"
        f"Business Question: {question}\n\n"
        f"User's Clarifications: {json.dumps(answers, indent=2)}\n\n"
        "Output the analysis plan as JSON."
    )
    return generate_json(prompt, system=PLANNING_SYSTEM_PROMPT)


def _slim_profile(profile: Dict) -> Dict:
    """Trim the profile so the prompt stays small and cheap.

    Keeps the structural summary and per-column dtype/null/unique info, but drops
    verbose ``top_values`` maps that are not needed for planning.
    """
    slim = {k: v for k, v in profile.items() if k != "columns"}
    slim_cols = {}
    for col, p in profile.get("columns", {}).items():
        slim_cols[col] = {
            "dtype": p.get("dtype"),
            "null_pct": p.get("null_pct"),
            "unique_count": p.get("unique_count"),
        }
    slim["columns"] = slim_cols
    return slim


def _validate_and_repair_plan(plan: Dict, profile: Dict) -> Dict[str, Any]:
    """Ensure the plan is internally consistent and references real columns."""
    columns = list(profile.get("columns", {}).keys())

    problem_type = plan.get("problem_type")
    if problem_type not in VALID_PROBLEM_TYPES:
        problem_type = "descriptive"

    target = plan.get("target_column")
    if target not in columns:
        target = None
    if problem_type == "clustering":
        target = None
    if problem_type in ("regression", "classification") and target is None:
        # Fall back to descriptive if we lost the target.
        problem_type = "descriptive"

    features = [
        c for c in (plan.get("feature_columns") or []) if c in columns and c != target
    ]
    if not features:
        features = [c for c in columns if c != target]

    analyses = [
        a for a in (plan.get("analyses_to_run") or []) if a in AVAILABLE_ANALYSES
    ]
    analyses = _enforce_analysis_rules(analyses, profile)

    return {
        "problem_type": problem_type,
        "target_column": target,
        "feature_columns": features,
        "analyses_to_run": analyses or ["distribution_analysis"],
        "clarifying_notes": str(plan.get("clarifying_notes", "")),
        "reasoning": str(plan.get("reasoning", "")),
        "planner": "claude",
    }


def _enforce_analysis_rules(analyses: List[str], profile: Dict) -> List[str]:
    """Drop analyses that don't apply to this dataset; add sensible defaults."""
    result = list(dict.fromkeys(analyses))  # dedupe, keep order
    if not profile.get("has_datetime") and "trend_decomposition" in result:
        result.remove("trend_decomposition")

    # segment_comparison needs a low-cardinality categorical column.
    has_segmentable = any(
        p.get("dtype") in ("categorical", "boolean") and p.get("unique_count", 99) < 10
        for p in profile.get("columns", {}).values()
    )
    if not has_segmentable and "segment_comparison" in result:
        result.remove("segment_comparison")

    # Always include distribution + correlation if numeric columns exist.
    if profile.get("numeric_columns"):
        for default in ("distribution_analysis", "correlation_analysis"):
            if default not in result:
                result.append(default)
    return result


def _heuristic_plan(profile: Dict, question: str) -> Dict[str, Any]:
    """Deterministic planner used when Gemini is unavailable."""
    columns = profile.get("columns", {})
    targets = profile.get("potential_target_columns", [])
    target = targets[0] if targets else None

    problem_type = "descriptive"
    if target is not None:
        tprof = columns.get(target, {})
        if tprof.get("dtype") == "numeric" and tprof.get("unique_count", 0) > 15:
            problem_type = "regression"
        else:
            problem_type = "classification"
    elif (
        not profile.get("potential_target_columns")
        and len(profile.get("numeric_columns", [])) >= 2
    ):
        problem_type = "clustering"

    features = [c for c in columns if c != target]
    analyses = _enforce_analysis_rules(["outlier_detection"], profile)

    return {
        "problem_type": problem_type,
        "target_column": target,
        "feature_columns": features,
        "analyses_to_run": analyses,
        "clarifying_notes": "Generated by the offline heuristic planner (no API key set).",
        "reasoning": (
            f"Chose '{problem_type}' based on the presence of target-like column "
            f"'{target}'."
            if target
            else f"No clear target column found; defaulted to '{problem_type}'."
        ),
        "planner": "heuristic",
    }
