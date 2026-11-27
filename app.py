"""Streamlit front end owned by Person 6; uses AnswerResponse as the backend contract.

Run:  streamlit run app.py

Two pages: "Analyse" (upload, profile, multi-turn chat) and "Traces" (trace dashboard).
The UI never computes analytics: answers come from ``run_agent``/the Orchestrator, and tables
and charts are displayed only from backend artifact files.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:  # allow `streamlit run app.py` without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from verisheet.config import get_settings  # noqa: E402
from verisheet.schemas import AnswerResponse, AnswerStatus, WorkbookProfile  # noqa: E402
from verisheet.ui_support import artifact_file, save_upload, status_style  # noqa: E402

PAGES = ("Analyse", "Traces")


# --------------------------------------------------------------------------------------------
# Backend access (cached per server process)
# --------------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Starting the agent…")
def _orchestrator():
    """Build the production orchestrator once; returns (orchestrator, error message)."""
    from verisheet.orchestrator import ComponentUnavailable, build_default_orchestrator

    try:
        return build_default_orchestrator(), None
    except ComponentUnavailable as exc:
        return None, f"The {exc} component is not available, so questions cannot be answered yet."
    except Exception as exc:  # configuration problems must not crash the page
        return None, f"The agent could not start ({type(exc).__name__}). Check `.env` (QWEN_*, EMBEDDING_*)."


def _ask(question: str) -> AnswerResponse:
    from verisheet.orchestrator import run_agent

    orchestrator, error = _orchestrator()
    if orchestrator is None:
        return AnswerResponse(status=AnswerStatus.FAILED, answer=error, trace_id="(not started)")
    return run_agent(question, str(st.session_state.workbook_path), st.session_state.session_id,
                     orchestrator=orchestrator)


# --------------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {"session_id": f"ui-{uuid.uuid4().hex[:12]}", "workbook_path": None, "workbook_name": None,
                "workbook_hash": None, "profile": None, "history": [], "page": PAGES[0],
                "trace_id": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _load_workbook(uploaded) -> None:
    """Quarantine + profile a new upload; a different workbook starts a fresh session."""
    from verisheet.profile import profile_workbook

    data = uploaded.getvalue()
    upload_root = Path(get_settings().artifact_dir) / "uploads"
    try:
        path = save_upload(uploaded.name, data, upload_root)
        profile = profile_workbook(path)
    except ValueError as exc:
        st.error(f"This file cannot be used: {exc}")
        return
    if profile.workbook_id != st.session_state.workbook_hash:
        st.session_state.update(workbook_path=path, workbook_name=uploaded.name,
                                workbook_hash=profile.workbook_id, profile=profile.model_dump(mode="json"),
                                history=[], session_id=f"ui-{uuid.uuid4().hex[:12]}")


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _render_profile(profile: WorkbookProfile) -> None:
    sheets = pd.DataFrame([{
        "sheet": s.name, "rows": s.row_count, "columns": len(s.columns), "hidden": s.hidden,
        "formula cells": s.formula_cell_count, "duplicate rows": s.duplicate_rows,
        "date ranges": ", ".join(f"{c}: {r.get('min')}…{r.get('max')}" for c, r in s.candidate_date_ranges.items()),
    } for s in profile.sheets])
    st.dataframe(sheets, hide_index=True, width="stretch")
    for sheet in profile.sheets:
        with st.expander(f"Columns · {sheet.name}"):
            st.dataframe(pd.DataFrame([{
                "column": c.name, "type": c.dtype, "roles": ", ".join(c.candidate_roles),
                "nulls": c.null_count, "unique": c.unique_count, "examples": ", ".join(c.safe_examples),
            } for c in sheet.columns]), hide_index=True, width="stretch")
    if profile.warnings:
        with st.expander(f"Quality warnings ({len(profile.warnings)})", expanded=True):
            st.dataframe(pd.DataFrame([{
                "severity": w.severity.value, "code": w.code, "sheet": w.sheet or "", "column": w.column or "",
                "message": w.message,
            } for w in profile.warnings]), hide_index=True, width="stretch")


def _render_answer(response: AnswerResponse, key: str) -> None:
    style = status_style(response.status)
    getattr(st, style.kind)(f"{style.icon} **{style.label}** — {style.meaning}")
    st.markdown(response.answer)
    if response.clarification_question:
        st.info(f"**Question for you:** {response.clarification_question}")
    if response.assumptions:
        with st.expander("Assumptions", expanded=response.status == AnswerStatus.VERIFIED):
            for item in response.assumptions:
                st.markdown(f"- {item}")
    root = get_settings().artifact_dir
    table = artifact_file(response.result_table_path, root)
    chart = artifact_file(response.chart_path, root)
    if table is not None and table.suffix.lower() == ".csv":
        st.dataframe(pd.read_csv(table), hide_index=True, width="stretch")
    if chart is not None:
        st.image(str(chart), caption=response.chart_spec.title if response.chart_spec else None)
    if response.verification is not None:
        report = response.verification
        with st.expander(f"Verification · {report.verification_id} · "
                         f"{'passed' if report.verified else 'did not pass'}"):
            st.markdown(f"Numeric checks: {'✅' if report.numeric_checks_passed else '❌'} · "
                        f"Chart checks: {'✅' if report.chart_checks_passed else '❌'} · "
                        f"Semantic checks: {'✅' if report.semantic_checks_passed else '❌'} · "
                        f"method: `{report.verification_method}`")
            if report.findings:
                st.dataframe(pd.DataFrame([f.model_dump(mode="json") for f in report.findings]),
                             hide_index=True, width="stretch")
    left, right = st.columns([4, 1])
    left.caption(f"Trace ID: `{response.trace_id}`")
    if response.trace_id.startswith("trace-") and right.button("Open trace", key=f"trace-{key}"):
        st.session_state.update(goto="Traces", trace_id=response.trace_id)  # applied before the radio
        st.rerun()


def _analyse_page() -> None:
    uploaded = st.file_uploader("Upload an XLSX or CSV workbook", type=["xlsx", "csv"])
    if uploaded is not None:
        _load_workbook(uploaded)
    if st.session_state.profile is None:
        st.info("Upload a workbook to start. Files are stored privately and checked before use.")
        return

    profile = WorkbookProfile.model_validate(st.session_state.profile)
    with st.expander(f"Workbook profile · {st.session_state.workbook_name} · "
                     f"{len(profile.sheets)} sheet(s)", expanded=not st.session_state.history):
        _render_profile(profile)

    for number, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            _render_answer(AnswerResponse.model_validate(turn["response"]), key=str(number))

    question = st.chat_input("Ask a question about this workbook (follow-ups keep verified context)")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Planning, executing and verifying…"):
            try:
                response = _ask(question)
            except Exception as exc:  # never show a stack trace to the user
                response = AnswerResponse(status=AnswerStatus.FAILED, trace_id="(unavailable)",
                                          answer=f"Something went wrong ({type(exc).__name__}). Please try again.")
        st.session_state.history.append({"question": question, "response": response.model_dump(mode="json")})
        st.rerun()


def render_app() -> None:
    """Render upload, workbook profile, chat, result/chart display and verification status.

    The UI should call the orchestrator service rather than reimplementing business logic.
    Retain session ID across turns; retain workbook reference until user replaces the upload.
    Display assumptions, clarification/refusal messages and trace ID. Render chart only when
    the backend provides a real chart artifact. Provide a link/expander to the trace dashboard.
    """
    st.set_page_config(page_title="VeriSheet", layout="wide")
    _init_state()
    if st.session_state.get("goto"):          # widget-backed keys may only change before render
        st.session_state.page = st.session_state.pop("goto")
    with st.sidebar:
        st.title("VeriSheet")
        st.caption("Agentic spreadsheet analysis with independent verification")
        st.radio("Page", PAGES, key="page")
        if st.session_state.workbook_name:
            st.markdown(f"**Workbook:** {st.session_state.workbook_name}")
            st.caption(f"Session `{st.session_state.session_id}`")
            if st.button("New conversation"):
                st.session_state.update(history=[], session_id=f"ui-{uuid.uuid4().hex[:12]}")
                st.rerun()
    if st.session_state.page == "Traces":
        from verisheet.observability.trace_dashboard import render_trace_dashboard

        settings = get_settings()
        render_trace_dashboard(settings.trace_dir, st.session_state.trace_id, settings.artifact_dir)
    else:
        st.title("Analyse a workbook")
        _analyse_page()


if __name__ == "__main__":
    render_app()
