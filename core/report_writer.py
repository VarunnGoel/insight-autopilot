"""Report writer — the second Claude call.

Takes all structured analysis + model + explanation results and produces a
stakeholder-ready business narrative in Markdown. When no API key is present, a
deterministic template builder assembles a solid report from the same data so
the app is fully usable offline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from core.confidence import confidence_to_label
from utils.formatters import titleize
from utils.llm import LLMError, generate_text, llm_is_available
from utils.logger import get_logger

log = get_logger(__name__)

REPORT_SYSTEM_PROMPT = """\
You are a senior data analyst writing a business intelligence report for a
non-technical audience. You receive a business question and structured analysis
results in JSON. Your job is to ANSWER the question directly.

Structure the report with these Markdown sections:
1. ## Executive Summary (start with the direct answer to the question, then 2-3 sentences of context)
2. ## Key Insights (one bullet cluster per analysis, in plain English)
3. ## Model Performance (what the model learned, in business terms)
4. ## Top Predictors (from feature importance, no jargon)
5. ## Caveats & Limitations (reference the confidence scores and data warnings)
6. ## Recommended Next Steps (2-3 concrete actions)

Rules:
- OPENING SENTENCE MUST DIRECTLY ANSWER THE USER'S QUESTION. Not "we analysed X".
- Be specific: say "Focus on the South region" not "consider focusing on areas of improvement".
- Never use statistical jargon without immediately explaining it.
- Give every number context.
- Explicitly flag any insight with confidence < 0.5 as preliminary.
- Use **bold** for emphasis only. NEVER use _underscore_ italics — underscores collide with
  variable names and numbers, causing formatting breaks.
- Output Markdown only (no JSON, no code fences).
"""


def generate_report(payload: Dict[str, Any]) -> Dict[str, str]:
    """Return {"markdown": str, "source": "claude"|"template"}."""
    if llm_is_available():
        try:
            prompt = _build_prompt(payload)
            md = generate_text(prompt, system=REPORT_SYSTEM_PROMPT, temperature=0.5)
            if md.strip():
                return {"markdown": md.strip(), "source": "claude"}
        except LLMError as exc:
            log.warning("Claude report failed, using template: %s", exc)
    return {"markdown": _template_report(payload), "source": "template"}


def _build_prompt(payload: Dict[str, Any]) -> str:
    # Strip private pipeline objects before serialising.
    clean = _strip_private(payload)
    return (
        f"Original Business Question: {payload.get('question', '')}\n\n"
        f"Analysis Plan:\n{json.dumps(clean.get('plan', {}), indent=2)}\n\n"
        f"Analysis Results:\n{json.dumps(clean.get('results', {}), indent=2, default=str)}\n\n"
        f"Model Results:\n{json.dumps(clean.get('model', {}), indent=2, default=str)}\n\n"
        f"Explanation:\n{json.dumps(clean.get('explanation', {}), indent=2, default=str)}\n\n"
        "Write the business intelligence report now."
    )


def _strip_private(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys starting with '_' (DataFrames, fitted pipelines) recursively."""

    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if not str(k).startswith("_")}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    return clean(payload)


# Offline template report


def _template_report(payload: Dict[str, Any]) -> str:
    question = payload.get("question", "the dataset")
    plan = payload.get("plan", {})
    results: Dict[str, Any] = payload.get("results", {})
    model: Dict[str, Any] = payload.get("model", {})
    explanation: Dict[str, Any] = payload.get("explanation", {})

    lines: List[str] = []
    lines.append("## Executive Summary")
    lines.append(
        f"This report analyses the dataset in response to the question: "
        f'*"{question}"*. The analysis approached this as a '
        f"**{plan.get('problem_type', 'descriptive')}** problem"
        + (
            f" targeting **{titleize(plan['target_column'])}**."
            if plan.get("target_column")
            else "."
        )
    )
    headline = _headline_finding(results, model)
    if headline:
        lines.append(headline)
    lines.append("")

    lines.append("## Key Insights")
    for atype, res in results.items():
        conf = res.get("confidence")
        label = confidence_to_label(conf) if conf is not None else ""
        tag = f" *({label}, {conf})*" if conf is not None else ""
        lines.append(f"**{titleize(atype)}**{tag}")
        lines.append(f"- {res.get('summary', '')}")
        for f in res.get("key_findings", [])[:4]:
            lines.append(f"- {f}")
        if conf is not None and conf < 0.5:
            lines.append("- ⚠️ *Preliminary — low confidence; interpret with caution.*")
        lines.append("")

    lines.append("## Model Performance")
    if model.get("trained"):
        if model["problem_type"] == "clustering":
            lines.append(
                f"Segmented the data into **{model['best_k']} clusters** "
                f"(silhouette score {model['silhouette_score']}, where higher means "
                f"cleaner separation)."
            )
        else:
            lines.append(
                f"Best model: **{model['best_model']}**, achieving a cross-validated "
                f"{model['metric_name']} of **{model['cv_mean']}** "
                f"(± {model['cv_std']}) across {model['n_samples']} rows. "
                f"The ± figure shows how much performance varied between data folds — "
                f"smaller is more reliable."
            )
    else:
        lines.append(
            f"*No predictive model was trained: {model.get('message', 'descriptive analysis only')}.*"
        )
    lines.append("")

    lines.append("## Top Predictors")
    plain = explanation.get("plain_english", []) if explanation.get("available") else []
    if plain:
        for sentence in plain:
            lines.append(f"- {sentence}")
    else:
        lines.append(
            "- Feature-level explanations are not available for this analysis."
        )
    lines.append("")

    lines.append("## Caveats & Limitations")
    caveats = _collect_caveats(results, model)
    if caveats:
        for c in caveats:
            lines.append(f"- {c}")
    else:
        lines.append("- No major data-quality issues were detected.")
    if plan.get("clarifying_notes"):
        lines.append(f"- {plan['clarifying_notes']}")
    lines.append("")

    lines.append("## Recommended Next Steps")
    for step in _next_steps(plan, model, results):
        lines.append(f"- {step}")
    lines.append("")
    lines.append("---")
    lines.append(
        "*Report generated by Insight Autopilot's offline template engine "
        "(set a KINTIO_API_KEY for an AI-written narrative).*"
    )
    return "\n".join(lines)


def _headline_finding(results: Dict[str, Any], model: Dict[str, Any]) -> str:
    if model.get("trained") and model.get("problem_type") in (
        "regression",
        "classification",
    ):
        return (
            f"The strongest signal came from the predictive model, which reached a "
            f"cross-validated {model['metric_name']} of {model['cv_mean']}."
        )
    corr = results.get("correlation_analysis", {})
    pairs = corr.get("data", {}).get("pairs", [])
    if pairs:
        top = pairs[0]
        return (
            f"The most notable relationship is between {titleize(top['columns'][0])} and "
            f"{titleize(top['columns'][1])} (correlation {top['correlation']})."
        )
    return ""


def _collect_caveats(results: Dict[str, Any], model: Dict[str, Any]) -> List[str]:
    caveats: List[str] = []
    for atype, res in results.items():
        for w in res.get("warnings", []):
            caveats.append(f"{titleize(atype)}: {w}")
        conf = res.get("confidence")
        if conf is not None and conf < 0.5:
            caveats.append(f"{titleize(atype)} has low confidence ({conf}).")
    if model.get("trained") and model.get("cv_std", 0) and model.get("cv_mean"):
        if model["cv_std"] > 0.15:
            caveats.append(
                "Model performance varied notably across folds — results are less stable."
            )
    return caveats


def _next_steps(
    plan: Dict[str, Any], model: Dict[str, Any], results: Dict[str, Any]
) -> List[str]:
    steps: List[str] = []
    if model.get("trained") and model.get("problem_type") != "clustering":
        steps.append(
            "Collect more labelled data for the target to further improve and validate the model."
        )
    if "outlier_detection" in results:
        steps.append(
            "Investigate the flagged outlier rows — they may be data errors or key edge cases."
        )
    if plan.get("problem_type") == "clustering":
        steps.append(
            "Give each discovered segment a business-friendly name and tailor actions per segment."
        )
    if not steps:
        steps.append(
            "Refine the business question and gather targeted data for a deeper follow-up analysis."
        )
    steps.append(
        "Share this report with stakeholders and confirm the findings against domain knowledge."
    )
    return steps[:3]
