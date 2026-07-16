"""Explainability layer (SHAP-free).

The architecture originally called for SHAP. To keep the project installable for
free on any Python version (SHAP pulls heavy native deps), we instead build the
explanations from scikit-learn's permutation importance plus each model's native
signal (tree ``feature_importances_`` or linear ``coef_``). We then translate the
numbers into plain-English sentences a non-technical stakeholder can read.

This gives the same *deliverable* — "which features matter and in which
direction, explained in words" — without the dependency cost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.formatters import titleize
from utils.logger import get_logger

log = get_logger(__name__)


def explain_model(model_result: Dict[str, Any]) -> Dict[str, Any]:
    """Produce global importance + plain-English explanations for a trained model.

    Returns:
        {
          "global_importance": [{"feature", "importance", "direction"}...],
          "top_features": [...],
          "plain_english": [str, ...],
          "sample_explanation": {...} | None,
        }
    """
    if not model_result.get("trained"):
        return {
            "available": False,
            "reason": model_result.get("message", "No model trained."),
        }

    if model_result["problem_type"] == "clustering":
        return _explain_clustering(model_result)

    pipe = model_result.get("_pipeline")
    X = model_result.get("_X")
    perm = model_result.get("feature_importances", [])

    directions = _feature_directions(pipe, X, model_result)
    global_importance = []
    for item in perm:
        feat = item["feature"]
        global_importance.append(
            {
                "feature": feat,
                "importance": item["importance"],
                "direction": directions.get(feat, "unclear"),
            }
        )

    top_features = [g["feature"] for g in global_importance[:5]]
    plain = _to_plain_english(global_importance[:5], model_result)
    sample = _sample_explanation(model_result, global_importance[:5])

    return {
        "available": True,
        "global_importance": global_importance,
        "top_features": top_features,
        "plain_english": plain,
        "sample_explanation": sample,
    }


def _feature_directions(pipe, X: pd.DataFrame, model_result: Dict) -> Dict[str, str]:
    """Estimate whether each raw feature pushes the prediction up or down.

    For numeric features we use the sign of the Pearson correlation between the
    feature and the (encoded) target — a simple, honest directional signal that
    works regardless of the underlying model.
    """
    directions: Dict[str, str] = {}
    y = model_result.get("_y")
    if X is None or y is None:
        return directions

    if model_result["problem_type"] == "classification":
        # Encode target to numeric codes for correlation sign only.
        y_num = pd.Series(pd.Categorical(y).codes, index=y.index)
    else:
        y_num = pd.to_numeric(y, errors="coerce")

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            sub = pd.concat([X[col], y_num], axis=1).dropna()
            if len(sub) > 3 and sub.iloc[:, 0].std() > 0:
                r = np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1]
                if not np.isnan(r):
                    directions[col] = "increases" if r > 0 else "decreases"
        else:
            directions[col] = "varies by category"
    return directions


def _to_plain_english(top: List[Dict[str, Any]], model_result: Dict) -> List[str]:
    target = titleize(model_result.get("target", "the outcome"))
    problem = model_result["problem_type"]
    sentences: List[str] = []
    for rank, item in enumerate(top, start=1):
        feat = titleize(item["feature"])
        direction = item["direction"]
        if direction in ("increases", "decreases"):
            verb = "raises" if direction == "increases" else "lowers"
            outcome = (
                "the predicted value"
                if problem == "regression"
                else f"the likelihood of {target}"
            )
            phrase = f"higher {feat} tends to {verb} {outcome}"
        elif direction == "varies by category":
            phrase = (
                f"different {feat} categories are associated with different outcomes"
            )
        else:
            phrase = f"{feat} is an influential factor"
        qualifier = "the strongest predictor" if rank == 1 else f"the #{rank} predictor"
        sentences.append(f"{feat} is {qualifier} — {phrase}.")
    return sentences


def _sample_explanation(
    model_result: Dict, top: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Explain one concrete row: show top features and how it compares to average."""
    X = model_result.get("_X")
    pipe = model_result.get("_pipeline")
    if X is None or pipe is None or len(X) == 0:
        return None
    row = X.iloc[[0]]
    try:
        pred = pipe.predict(row)[0]
    except Exception:
        return None

    contributions = []
    for item in top:
        feat = item["feature"]
        if pd.api.types.is_numeric_dtype(X[feat]):
            val = row[feat].iloc[0]
            avg = X[feat].mean()
            comparison = "above" if val > avg else "below"
            contributions.append(
                {
                    "feature": feat,
                    "value": None if pd.isna(val) else round(float(val), 3),
                    "vs_average": comparison,
                    "direction": item["direction"],
                }
            )
        else:
            contributions.append(
                {
                    "feature": feat,
                    "value": str(row[feat].iloc[0]),
                    "vs_average": "n/a",
                    "direction": item["direction"],
                }
            )
    return {
        "prediction": (
            str(pred)
            if not isinstance(pred, (int, float, np.floating))
            else round(float(pred), 3)
        ),
        "contributions": contributions,
    }


def _explain_clustering(model_result: Dict) -> Dict[str, Any]:
    profiles = model_result.get("cluster_profiles", {})
    numeric = model_result.get("_numeric", [])
    plain: List[str] = []
    for cluster_id, means in profiles.items():
        # Describe each cluster by its most distinctive feature.
        if not means:
            continue
        # Pick the feature furthest from the overall mean across clusters.
        sorted_feats = sorted(means.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_feat = (
            sorted_feats[0][0]
            if sorted_feats
            else (numeric[0] if numeric else "features")
        )
        plain.append(
            f"Cluster {cluster_id} ({model_result['cluster_sizes'].get(int(cluster_id), '?')} rows) "
            f"is characterised most by its {titleize(top_feat)} level "
            f"(avg {means.get(top_feat, 'n/a')})."
        )
    return {
        "available": True,
        "global_importance": [],
        "top_features": numeric[:5],
        "plain_english": plain,
        "sample_explanation": None,
    }
