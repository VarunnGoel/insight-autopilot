"""Automatic ML model selection and training.

Based on the agent's ``problem_type``, this module builds a leak-free sklearn
preprocessing + model Pipeline, cross-validates several candidate algorithms,
and returns the best one with its metrics and feature importances.

We deliberately stick to scikit-learn (no XGBoost/SHAP) so the whole project
installs cleanly on any Python version with zero native build steps, while still
demonstrating the professional patterns interviewers look for: ColumnTransformer
pipelines, cross-validation, and permutation-based feature importance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from utils.config import settings
from utils.logger import get_logger

log = get_logger(__name__)


def _looks_like_id(df: pd.DataFrame, col: str) -> bool:
    """Heuristic: integer-like column that is (almost) unique per row -> an ID."""
    series = df[col].dropna()
    if len(series) < 20:
        return False
    name = col.lower()
    nearly_unique = series.nunique() >= 0.95 * len(series)
    integer_like = pd.api.types.is_integer_dtype(series) or (
        pd.api.types.is_numeric_dtype(series) and (series % 1 == 0).all()
    )
    return nearly_unique and (integer_like or name == "id" or name.endswith("_id"))


def _split_feature_types(
    df: pd.DataFrame, features: List[str]
) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for col in features:
        if _looks_like_id(df, col):
            continue  # IDs carry no predictive signal and cause overfitting
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            # Skip very high-cardinality text columns — they hurt more than help.
            if df[col].nunique(dropna=True) <= 30:
                categorical.append(col)
    return numeric, categorical


def _build_preprocessor(numeric: List[str], categorical: List[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        # sparse_output kwarg name changed across sklearn versions; handle both.
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # older sklearn
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", ohe),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def select_and_train(df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the right training routine based on the plan's problem type."""
    problem_type = plan.get("problem_type", "descriptive")
    if problem_type == "regression":
        return _train_regression(df, plan)
    if problem_type == "classification":
        return _train_classification(df, plan)
    if problem_type == "clustering":
        return _train_clustering(df, plan)
    return {
        "problem_type": "descriptive",
        "trained": False,
        "message": "No predictive model trained (descriptive analysis).",
    }


def _prepare_xy(
    df: pd.DataFrame, plan: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    target = plan["target_column"]
    features = [c for c in plan["feature_columns"] if c != target]
    work = df[features + [target]].copy()
    work = work.dropna(subset=[target])
    if len(work) > settings.max_rows_for_modeling:
        work = work.sample(
            settings.max_rows_for_modeling, random_state=settings.random_state
        )
    X = work[features]
    y = work[target]
    numeric, categorical = _split_feature_types(work, features)
    return X, y, numeric, categorical


def _train_regression(df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline

    X, y, numeric, categorical = _prepare_xy(df, plan)
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(X) < 30 or not (numeric or categorical):
        return {
            "problem_type": "regression",
            "trained": False,
            "message": "Not enough clean rows/features to train a regression model.",
        }

    pre = _build_preprocessor(numeric, categorical)
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=settings.random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=settings.random_state, n_jobs=-1
        ),
    }
    cv = KFold(
        n_splits=min(settings.cv_splits, max(2, len(X) // 10)),
        shuffle=True,
        random_state=settings.random_state,
    )

    scored = {}
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        try:
            r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            scored[name] = r2
        except Exception as exc:
            log.warning("Regressor %s failed CV: %s", name, exc)

    if not scored:
        return {
            "problem_type": "regression",
            "trained": False,
            "message": "All regression candidates failed cross-validation.",
        }

    best_name = max(scored, key=lambda k: scored[k].mean())
    best_pipe = Pipeline([("pre", pre), ("model", candidates[best_name])]).fit(X, y)
    importances = _permutation_importance(best_pipe, X, y, "r2")

    return {
        "problem_type": "regression",
        "trained": True,
        "best_model": best_name,
        "metric_name": "R²",
        "cv_mean": round(float(scored[best_name].mean()), 4),
        "cv_std": round(float(scored[best_name].std()), 4),
        "cv_scores": {k: [round(float(s), 4) for s in v] for k, v in scored.items()},
        "feature_importances": importances,
        "n_samples": int(len(X)),
        "target": plan["target_column"],
        "_pipeline": best_pipe,
        "_X": X,
        "_y": y,
        "_numeric": numeric,
        "_categorical": categorical,
    }


def _train_classification(df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    X, y, numeric, categorical = _prepare_xy(df, plan)
    y = y.astype("category")
    if y.nunique() < 2 or len(X) < 30 or not (numeric or categorical):
        return {
            "problem_type": "classification",
            "trained": False,
            "message": "Not enough clean rows/classes/features to train a classifier.",
        }
    # Drop ultra-rare classes that break StratifiedKFold.
    counts = y.value_counts()
    keep = counts[counts >= 5].index
    mask = y.isin(keep)
    X, y = X[mask], y[mask].astype("category")
    if y.nunique() < 2:
        return {
            "problem_type": "classification",
            "trained": False,
            "message": "After removing rare classes, fewer than 2 classes remain.",
        }

    pre = _build_preprocessor(numeric, categorical)
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=settings.random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    n_splits = min(settings.cv_splits, int(y.value_counts().min()))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=settings.random_state
    )

    scoring = "f1_weighted"
    scored = {}
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        try:
            f1 = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
            scored[name] = f1
        except Exception as exc:
            log.warning("Classifier %s failed CV: %s", name, exc)

    if not scored:
        return {
            "problem_type": "classification",
            "trained": False,
            "message": "All classification candidates failed cross-validation.",
        }

    best_name = max(scored, key=lambda k: scored[k].mean())
    best_pipe = Pipeline([("pre", pre), ("model", candidates[best_name])]).fit(X, y)
    importances = _permutation_importance(best_pipe, X, y, scoring)

    return {
        "problem_type": "classification",
        "trained": True,
        "best_model": best_name,
        "metric_name": "F1 (weighted)",
        "cv_mean": round(float(scored[best_name].mean()), 4),
        "cv_std": round(float(scored[best_name].std()), 4),
        "cv_scores": {k: [round(float(s), 4) for s in v] for k, v in scored.items()},
        "feature_importances": importances,
        "n_classes": int(y.nunique()),
        "classes": [str(c) for c in y.cat.categories],
        "n_samples": int(len(X)),
        "target": plan["target_column"],
        "_pipeline": best_pipe,
        "_X": X,
        "_y": y,
        "_numeric": numeric,
        "_categorical": categorical,
    }


def _train_clustering(df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = plan["feature_columns"]
    numeric, categorical = _split_feature_types(df, features)
    if len(numeric) < 2:
        return {
            "problem_type": "clustering",
            "trained": False,
            "message": "Clustering needs at least 2 numeric features.",
        }

    pre = _build_preprocessor(numeric, categorical)
    X = df[numeric + categorical].dropna()
    if len(X) < 20:
        return {
            "problem_type": "clustering",
            "trained": False,
            "message": "Not enough complete rows to cluster.",
        }
    X_trans = pre.fit_transform(X)

    best_k, best_score, best_labels = None, -1.0, None
    scores_by_k = {}
    for k in range(2, min(9, len(X) // 2 + 1)):
        km = KMeans(n_clusters=k, random_state=settings.random_state, n_init=10)
        labels = km.fit_predict(X_trans)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_trans, labels)
        scores_by_k[k] = round(float(score), 4)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_k is None:
        return {
            "problem_type": "clustering",
            "trained": False,
            "message": "Could not find a valid clustering.",
        }

    # Profile each cluster by its mean on the numeric features.
    profiled = X.copy()
    profiled["_cluster"] = best_labels
    cluster_profiles = (
        profiled.groupby("_cluster")[numeric].mean().round(3).to_dict(orient="index")
    )
    cluster_sizes = pd.Series(best_labels).value_counts().to_dict()

    return {
        "problem_type": "clustering",
        "trained": True,
        "best_k": best_k,
        "metric_name": "Silhouette",
        "silhouette_score": round(float(best_score), 4),
        "scores_by_k": scores_by_k,
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
        "cluster_profiles": {int(k): v for k, v in cluster_profiles.items()},
        "n_samples": int(len(X)),
        "_labels": best_labels.tolist(),
        "_numeric": numeric,
    }


def _permutation_importance(
    pipe, X: pd.DataFrame, y: pd.Series, scoring: str
) -> List[Dict[str, Any]]:
    """Model-agnostic feature importance via permutation (a SHAP-free stand-in)."""
    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(
            pipe,
            X,
            y,
            scoring=scoring,
            n_repeats=8,
            random_state=settings.random_state,
            n_jobs=-1,
        )
    except Exception as exc:
        log.warning("Permutation importance failed: %s", exc)
        return []

    importances = [
        {"feature": col, "importance": round(float(imp), 5)}
        for col, imp in zip(X.columns, result.importances_mean)
    ]
    importances.sort(key=lambda d: d["importance"], reverse=True)
    return importances
