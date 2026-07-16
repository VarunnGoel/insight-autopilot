"""Plotly chart builders for the dashboard.

Each function returns a ``plotly.graph_objects.Figure`` (or None when there is
nothing to plot) so ``app.py`` stays declarative: build a figure, hand it to
``st.plotly_chart``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.formatters import titleize

_TEMPLATE = "plotly_white"


def column_type_bar(profile: Dict[str, Any]) -> Optional[go.Figure]:
    counts: Dict[str, int] = {}
    for p in profile.get("columns", {}).values():
        counts[p["dtype"]] = counts.get(p["dtype"], 0) + 1
    if not counts:
        return None
    fig = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        labels={"x": "Column type", "y": "Count"},
        title="Column Type Breakdown",
        template=_TEMPLATE,
        color=list(counts.keys()),
    )
    fig.update_layout(showlegend=False)
    return fig


def null_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty:
        return None
    null_pct = df.isna().mean().sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        return None
    fig = px.bar(
        x=[titleize(c) for c in null_pct.index],
        y=(null_pct.values * 100),
        title="Missing Values by Column",
        template=_TEMPLATE,
        labels={"x": "Column", "y": "% missing"},
        color=(null_pct.values * 100),
        color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def distribution_hist(df: pd.DataFrame, column: str) -> Optional[go.Figure]:
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return None
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Distribution of {titleize(column)}",
        template=_TEMPLATE,
        marginal="box",
    )
    return fig


def correlation_heatmap(results: Dict[str, Any]) -> Optional[go.Figure]:
    corr = results.get("correlation_analysis", {}).get("data", {}).get("matrix")
    if not corr:
        return None
    matrix = pd.DataFrame(corr)
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
        template=_TEMPLATE,
    )
    return fig


def outlier_scatter(
    df: pd.DataFrame, results: Dict[str, Any], numeric_cols: List[str]
) -> Optional[go.Figure]:
    data = results.get("outlier_detection", {}).get("data", {})
    idx = set(data.get("outlier_indices", []))
    if len(numeric_cols) < 2:
        return None
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    plot_df = df[[x_col, y_col]].dropna().copy()
    plot_df["status"] = ["Outlier" if i in idx else "Normal" for i in plot_df.index]
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="status",
        color_discrete_map={"Outlier": "#d73027", "Normal": "#4575b4"},
        title=f"Outliers: {titleize(x_col)} vs {titleize(y_col)}",
        template=_TEMPLATE,
    )
    return fig


def feature_importance_bar(model: Dict[str, Any]) -> Optional[go.Figure]:
    importances = model.get("feature_importances", [])
    if not importances:
        return None
    top = importances[:12][::-1]  # reverse for horizontal bar readability
    fig = px.bar(
        x=[i["importance"] for i in top],
        y=[titleize(i["feature"]) for i in top],
        orientation="h",
        title="Feature Importance (permutation)",
        labels={"x": "Importance", "y": "Feature"},
        template=_TEMPLATE,
    )
    return fig


def cv_scores_box(model: Dict[str, Any]) -> Optional[go.Figure]:
    cv = model.get("cv_scores", {})
    if not cv:
        return None
    fig = go.Figure()
    for name, scores in cv.items():
        fig.add_trace(go.Box(y=scores, name=name, boxpoints="all"))
    fig.update_layout(
        title=f"Cross-Validated {model.get('metric_name', 'Score')} by Model",
        template=_TEMPLATE,
        yaxis_title=model.get("metric_name", "Score"),
    )
    return fig


def trend_line(results: Dict[str, Any]) -> Optional[go.Figure]:
    data = results.get("trend_decomposition", {}).get("data", {})
    if not data.get("dates"):
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["dates"],
            y=data["values"],
            mode="lines",
            name="Actual",
            line=dict(color="#4575b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["dates"],
            y=data["trend"],
            mode="lines",
            name="Trend",
            line=dict(color="#d73027", width=3),
        )
    )
    fig.update_layout(title="Trend Decomposition", template=_TEMPLATE)
    return fig


def cluster_sizes_bar(model: Dict[str, Any]) -> Optional[go.Figure]:
    sizes = model.get("cluster_sizes", {})
    if not sizes:
        return None
    fig = px.bar(
        x=[f"Cluster {k}" for k in sizes.keys()],
        y=list(sizes.values()),
        title="Cluster Sizes",
        labels={"x": "Cluster", "y": "Rows"},
        template=_TEMPLATE,
        color=[f"Cluster {k}" for k in sizes.keys()],
    )
    fig.update_layout(showlegend=False)
    return fig
