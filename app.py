"""Insight Autopilot — Streamlit dashboard entry point.

Keeps the UI thin: all real logic lives in ``core/``. This file wires together
upload/config, the analysis run, and the results tabs, and persists finished runs
to SQLite for the session-history sidebar.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core import ingestion, storage
from core.confidence import confidence_color, confidence_to_label
from core.pipeline import run_full_analysis
from ui import charts
from ui.report_renderer import render_html
from utils.config import settings
from utils.formatters import titleize

st.set_page_config(page_title="Insight Autopilot", page_icon="🔮", layout="wide")


# Session state helpers


def _init_state() -> None:
    defaults = {
        "df": None,
        "dataset_name": None,
        "question": "",
        "stage": "upload",
        "output": None,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def _reset() -> None:
    for key in ("df", "dataset_name", "question", "output"):
        st.session_state[key] = None
    st.session_state["stage"] = "upload"


# Sidebar


def _sidebar() -> None:
    with st.sidebar:
        st.header("🔮 Insight Autopilot")
        st.caption("Agentic BI assistant powered by Claude AI.")

        if settings.llm_available:
            st.success(f"Claude connected · `{settings.claude_model}`")
        else:
            st.warning(
                "No `KINTIO_API_KEY` set — running in **offline mode** "
                "(heuristic planner + template report). Add a key in `.env` "
                "for AI-written plans and reports."
            )

        st.divider()
        st.subheader("Recent sessions")
        try:
            sessions = storage.list_sessions(limit=5)
        except Exception:
            sessions = []
        if not sessions:
            st.caption("No saved sessions yet.")
        for s in sessions:
            label = f"{s['dataset_name']} · {s['problem_type']}"
            if st.button(label, key=f"load_{s['id']}", use_container_width=True):
                loaded = storage.load_session(s["id"])
                if loaded:
                    st.info("Loaded session metadata. Re-run to regenerate charts.")
                    st.json(
                        {k: loaded[k] for k in ("business_question", "problem_type")}
                    )

        st.divider()
        if st.button("🔄 Start over", use_container_width=True):
            _reset()
            st.rerun()

        with st.expander("About"):
            st.markdown(
                "- Profiles any CSV/Excel dataset\n"
                "- Uses an AI agent to plan the analysis\n"
                "- Auto-selects & trains an ML model\n"
                "- Scores confidence on every insight\n"
                "- Writes a stakeholder report"
            )


# Stage 1: Upload & configure


def _stage_upload() -> None:
    st.title("Upload data & ask a question")

    col1, col2 = st.columns([3, 2])
    with col1:
        uploaded = st.file_uploader(
            "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"]
        )
        st.markdown("**— or —**")
        sample_files = settings.sample_files()
        sample_names = ["(none)"] + [p.name for p in sample_files]
        chosen = st.selectbox("Try a sample dataset", sample_names)

    with col2:
        st.markdown("#### Example questions")
        st.markdown(
            "- *What predicts customer churn?*\n"
            "- *Why are sales dropping over time?*\n"
            "- *What natural segments exist in my customers?*"
        )

    df = None
    dataset_name = None
    if uploaded is not None:
        try:
            loaded = ingestion.load_any(uploaded, filename=uploaded.name)
            df, dataset_name = loaded["df"], uploaded.name
            _show_validation(loaded["validation"])
        except Exception as exc:
            st.error(f"Could not load file: {exc}")
    elif chosen != "(none)":
        path = settings.sample_data_dir / chosen
        loaded = ingestion.load_any(str(path), filename=chosen)
        df, dataset_name = loaded["df"], chosen
        _show_validation(loaded["validation"])

    if df is not None:
        st.session_state["df"] = df
        st.session_state["dataset_name"] = dataset_name

        tab_preview, tab_ask = st.tabs(["📋 Preview data", "❓ Ask a question"])
        with tab_preview:
            st.dataframe(df.head(10), use_container_width=True)
        with tab_ask:
            question = st.text_input(
                "What do you want to know?",
                value=st.session_state.get("question", ""),
                placeholder="e.g. What are the strongest drivers of customer churn?",
                label_visibility="collapsed",
            )
            if st.button(
                "🚀 Analyse",
                type="primary",
                use_container_width=True,
                disabled=len(question.strip()) < 5,
            ):
                st.session_state["question"] = question
                st.session_state["stage"] = "run"
                st.rerun()


def _show_validation(validation: dict) -> None:
    for err in validation.get("errors", []):
        st.error(err)
    for warn in validation.get("warnings", []):
        st.warning(warn)


# Stage 2: Run


def _stage_run() -> None:
    st.title("Running analysis")
    df = st.session_state["df"]
    question = st.session_state["question"]
    progress = st.progress(0.0)
    status = st.empty()

    def _cb(fraction: float, message: str) -> None:
        progress.progress(min(1.0, fraction))
        status.info(message)

    try:
        output = run_full_analysis(df, question, progress=_cb)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        if st.button("Back to start"):
            _reset()
            st.rerun()
        return

    st.session_state["output"] = output

    # Persist the run (best-effort — ignore storage errors on ephemeral FS).
    try:
        storage.save_session(
            dataset_name=st.session_state["dataset_name"] or "dataset",
            row_count=int(df.shape[0]),
            col_count=int(df.shape[1]),
            business_question=question,
            plan=output["plan"],
            results=output["results"],
            report_text=output["report"]["markdown"],
            model_performance=output["model"],
        )
    except Exception:
        pass

    st.session_state["stage"] = "results"
    st.rerun()


# Stage 4: Results


def _stage_results() -> None:
    output = st.session_state["output"]
    df = st.session_state["df"]
    profile, plan = output["profile"], output["plan"]
    results, model, explanation = (
        output["results"],
        output["model"],
        output["explanation"],
    )

    st.title("Results")
    planner = plan.get("planner", "claude")
    st.caption(
        f"Problem type: **{plan['problem_type']}** · "
        f"Planner: **{planner}** · Analyses: {', '.join(plan['analyses_to_run'])}"
    )
    if plan.get("reasoning"):
        st.info(f"**Agent reasoning:** {plan['reasoning']}")

    # Top metric row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{profile['shape']['rows']:,}")
    m2.metric("Columns", profile["shape"]["cols"])
    m3.metric("Problem type", plan["problem_type"].title())
    if model.get("trained") and model["problem_type"] != "clustering":
        m4.metric(model["metric_name"], model["cv_mean"])
    elif model.get("trained"):
        m4.metric("Clusters", model.get("best_k", "—"))
    else:
        m4.metric("Model", "—")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data Profile", "💡 Key Insights", "🤖 ML Model", "📄 Full Report"]
    )

    with tab1:
        _render_profile_tab(df, profile)
    with tab2:
        _render_insights_tab(df, results, profile)
    with tab3:
        _render_model_tab(model, explanation)
    with tab4:
        _render_report_tab(output)


def _render_profile_tab(df: pd.DataFrame, profile: dict) -> None:
    c1, c2 = st.columns(2)
    fig = charts.column_type_bar(profile)
    if fig:
        c1.plotly_chart(fig, use_container_width=True)
    fig = charts.null_heatmap(df)
    if fig:
        c2.plotly_chart(fig, use_container_width=True)

    numeric_cols = profile.get("numeric_columns", [])
    if numeric_cols:
        col = st.selectbox("Distribution of column", numeric_cols)
        fig = charts.distribution_hist(df, col)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def _render_insights_tab(df: pd.DataFrame, results: dict, profile: dict) -> None:
    if not results:
        st.info("No analyses were run.")
        return
    for atype, res in results.items():
        conf = res.get("confidence")
        with st.container(border=True):
            head_cols = st.columns([4, 1])
            head_cols[0].subheader(titleize(atype))
            if conf is not None:
                color = confidence_color(conf)
                head_cols[1].markdown(
                    f"<div style='text-align:right'><span style='background:{color};"
                    f"color:white;padding:4px 10px;border-radius:12px;font-size:0.8em'>"
                    f"{confidence_to_label(conf)} · {conf}</span></div>",
                    unsafe_allow_html=True,
                )
            st.write(res.get("summary", ""))
            for finding in res.get("key_findings", []):
                st.markdown(f"- {finding}")
            for warn in res.get("warnings", []):
                st.caption(f"⚠️ {warn}")

    st.divider()
    c1, c2 = st.columns(2)
    fig = charts.correlation_heatmap(results)
    if fig:
        c1.plotly_chart(fig, use_container_width=True)
    fig = charts.outlier_scatter(df, results, profile.get("numeric_columns", []))
    if fig:
        c2.plotly_chart(fig, use_container_width=True)
    fig = charts.trend_line(results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def _render_model_tab(model: dict, explanation: dict) -> None:
    if not model.get("trained"):
        st.info(
            model.get("message", "No predictive model was trained for this problem.")
        )
        return

    if model["problem_type"] == "clustering":
        c1, c2 = st.columns(2)
        c1.metric("Clusters found", model["best_k"])
        c2.metric("Silhouette score", model["silhouette_score"])
        fig = charts.cluster_sizes_bar(model)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Cluster profiles (feature averages)")
        st.dataframe(
            pd.DataFrame(model["cluster_profiles"]).T, use_container_width=True
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", model["best_model"])
        c2.metric(model["metric_name"], model["cv_mean"])
        c3.metric("± Std (fold variation)", model["cv_std"])
        fig = charts.feature_importance_bar(model)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        fig = charts.cv_scores_box(model)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if explanation.get("available") and explanation.get("plain_english"):
        st.subheader("What drives the predictions")
        for sentence in explanation["plain_english"]:
            st.markdown(f"- {sentence}")


def _render_report_tab(output: dict) -> None:
    report = output["report"]
    source = report.get("source", "template")
    st.caption(
        f"Report written by: **{'Claude AI' if source == 'claude' else 'offline template engine'}**"
    )
    st.markdown(report["markdown"])

    html = render_html(
        report["markdown"], question=st.session_state.get("question", "")
    )
    st.download_button(
        "⬇️ Download report as HTML",
        data=html,
        file_name=f"insight_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
    )


# Router


def main() -> None:
    _init_state()
    _sidebar()
    stage = st.session_state["stage"]
    if stage == "upload":
        _stage_upload()
    elif stage == "run":
        _stage_run()
    elif stage == "results":
        _stage_results()


if __name__ == "__main__":
    main()
