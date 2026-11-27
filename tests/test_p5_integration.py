"""End-to-end P5 integration with the real P2 profiler, P3 Planner and P4 Executor.

Only the Qwen endpoint is faked (scripted JSON replies), so these tests exercise the actual
plan validation, typed execution, chart rendering, independent verification, critic routing,
state persistence and tracing that the live demo uses.
"""

from __future__ import annotations

import json

import pytest

import verisheet.executor as executor_mod
from p5_support import FakeQwen, ListTraceLogger, write_retail_csv, write_superstore
from verisheet.critic import Critic, CriticMode
from verisheet.executor import Executor
from verisheet.orchestrator import AgentOrchestrator
from verisheet.planner import Planner
from verisheet.profile import profile_workbook
from verisheet.schemas import AnswerStatus, ChartType, LLMResponse
from verisheet.state import StateStore

ACCEPT = json.dumps({"decision": "ACCEPT", "intent_match": True, "chart_appropriate": True,
                     "rationale": "Answers the question."})


class ScriptedQwen:
    """QwenClient stand-in for the Planner: returns queued plan dicts as JSON."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system_prompt, user_prompt, temperature=0.0, max_tokens=None):
        self.calls.append({"user": user_prompt})
        reply = self.replies.pop(0)
        return LLMResponse(content=reply if isinstance(reply, str) else json.dumps(reply), model="Qwen/Qwen3-8B",
                           input_tokens=1200, output_tokens=300, latency_ms=900.0, finish_reason="stop")


def quarter_reply(chart="bar"):
    return {
        "plan_id": "x", "goal": "compare_profit_last_two_completed_quarters",
        "metric": {"name": "profit", "source_columns": ["Profit"], "aggregation": "sum", "formula": "sum(Profit)"},
        "source_sheets": ["Orders"], "date_policy": "previous_to_quarter_containing_max_observed_date",
        "steps": [
            {"step_id": "quarter-profit", "tool": "run_analysis", "operation": "quarter_comparison",
             "inputs": {"sheet": "Orders", "date_column": "Order_Date", "metric_column": "Profit", "aggregation": "sum"}},
            {"step_id": "chart", "tool": "make_chart", "depends_on": ["quarter-profit"],
             "inputs": {"source_step_id": "quarter-profit"}}],
        "chart_spec": {"chart_type": chart, "title": "Profit: last two completed quarters",
                       "x_column": "quarter", "y_column": "Profit"},
        "assumptions": ["Last completed quarter = quarter before the quarter containing the max observed "
                        "Order_Date (no Data_Through field)."]}


def region_reply(agg="sum", chart="bar"):
    return {
        "plan_id": "x", "goal": "sales_by_region",
        "metric": {"name": "sales", "source_columns": ["Sales"], "aggregation": agg, "formula": f"{agg}(Sales)"},
        "source_sheets": ["Orders"], "group_by": ["Region"],
        "steps": [
            {"step_id": "by-region", "tool": "run_analysis", "operation": "group_aggregate",
             "inputs": {"sheet": "Orders", "group_by": ["Region"], "metrics": {"Sales": agg}}},
            {"step_id": "chart", "tool": "make_chart", "depends_on": ["by-region"],
             "inputs": {"source_step_id": "by-region"}}],
        "chart_spec": {"chart_type": chart, "title": "Sales by region", "x_column": "Region", "y_column": "Sales"},
        "assumptions": []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep artifacts/traces inside the test directory
    workbook = str(write_superstore(tmp_path / "superstore.xlsx"))
    artifacts = tmp_path / "artifacts"
    return {"wb": workbook, "artifacts": artifacts, "tmp": tmp_path, "trace": ListTraceLogger()}


def build(env, planner_client, critic=None, **kw) -> AgentOrchestrator:
    return AgentOrchestrator(
        profiler=profile_workbook, retriever=None, planner=Planner(client=planner_client),
        executor=Executor(artifact_dir=str(env["artifacts"])),
        critic=critic or Critic(mode=CriticMode.DETERMINISTIC),
        state_store=StateStore(str(env["tmp"] / "sessions")), trace_logger=env["trace"],
        artifact_roots=[env["artifacts"]], **kw)


def test_live_demo_flow_verified_then_follow_up_keeps_context(env):
    qwen = ScriptedQwen(quarter_reply(), region_reply())
    critic_llm = FakeQwen(ACCEPT, ACCEPT)
    orch = build(env, qwen, Critic(client=critic_llm))
    first = orch.answer_question("What was recorded profit in the last fully completed quarter, "
                                 "compared with the previous quarter?", env["wb"], "demo")
    assert first.status == AnswerStatus.VERIFIED, first.answer
    assert "2024Q3" in first.answer and "130.00" in first.answer and "40.00" in first.answer
    assert first.chart_path and first.chart_path.endswith(".png")
    assert any("Order_Date" in a for a in first.assumptions)
    planner_event = next(e for e in env["trace"].events if e.actor == "planner")
    assert planner_event.input_tokens == 1200 and planner_event.prompt_version and planner_event.model_name
    # Follow-up: the Planner prompt now carries the verified quarter.
    second = orch.answer_question("Break total sales down by region.", env["wb"], "demo")
    assert second.status == AnswerStatus.VERIFIED
    assert "2024Q3" in qwen.calls[1]["user"]
    # The follow-up keeps the verified quarter (VS-P3-04/P5-04): 2024Q3 sales, not all-time
    # totals (East 1,600 / West 2,000), which the earlier assertion had locked in by mistake.
    assert "East: 700" in second.answer and "West: 600" in second.answer


def test_injected_plot_value_fault_is_repaired_by_rerendering_only_the_chart(env, monkeypatch):
    real_make_chart = executor_mod.make_chart
    calls = {"n": 0}

    def faulty_make_chart(path, spec, out):
        calls["n"] += 1
        result = real_make_chart(path, spec, out)
        if calls["n"] == 1:  # deliberate plot-value fault on the first render
            y = spec.y_column
            result["plotted_data"][y] = [v + 999 for v in result["plotted_data"][y]]
        return result

    monkeypatch.setattr(executor_mod, "make_chart", faulty_make_chart)
    orch = build(env, ScriptedQwen(quarter_reply()))
    response = orch.answer_question("Profit in the last completed quarter vs the previous one?", env["wb"], "s")
    assert response.status == AnswerStatus.VERIFIED and calls["n"] == 2
    transitions = [(e.status, e.failure_category) for e in orch.last_events if e.event_type == "transition"]
    assert ("REPAIR", "chart_value_mismatch") in transitions
    reruns = [e.step_id for e in orch.last_events if e.event_type == "execute_step"]
    assert reruns == ["quarter-profit", "chart", "chart"]


def test_injected_wrong_aggregation_is_detected_and_repaired(env, monkeypatch):
    real = executor_mod.run_analysis
    calls = {"n": 0}

    def faulty(operation, frames, inputs):
        calls["n"] += 1
        if calls["n"] == 1:
            inputs = {**inputs, "metrics": {k: "mean" for k in inputs["metrics"]}}
        return real(operation, frames, inputs)

    monkeypatch.setattr(executor_mod, "run_analysis", faulty)
    orch = build(env, ScriptedQwen(region_reply()))
    response = orch.answer_question("What were total sales by region?", env["wb"], "s")
    assert response.status == AnswerStatus.VERIFIED
    critic_events = [e for e in orch.last_events if e.actor == "critic"]
    assert critic_events[0].status == "REPAIR" and critic_events[0].failure_category == "wrong_aggregation"
    assert "mean(Sales)" in critic_events[0].output_summary


def test_plan_that_averages_a_total_is_replanned_before_execution(env):
    qwen = ScriptedQwen(region_reply(agg="mean"), region_reply(agg="sum"))
    orch = build(env, qwen)
    response = orch.answer_question("What were total sales by region?", env["wb"], "s")
    assert response.status == AnswerStatus.VERIFIED
    assert len(qwen.calls) == 2 and "wrong_metric" in qwen.calls[1]["user"]
    executed = [e for e in orch.last_events if e.event_type == "execute_step"]
    assert len(executed) == 2  # only the revised plan ran


def test_ranking_drawn_as_line_is_fixed_without_replanning(env):
    reply = region_reply(chart="line")
    reply["goal"] = "regions_with_highest_sales"
    qwen = ScriptedQwen(reply)
    response = build(env, qwen).answer_question("Which region has the highest total sales?", env["wb"], "s")
    assert response.status == AnswerStatus.VERIFIED and len(qwen.calls) == 1
    assert response.chart_spec.chart_type == ChartType.HORIZONTAL_BAR


def test_online_retail_profit_is_a_clarification_not_a_number(env):
    csv = str(write_retail_csv(env["tmp"] / "retail.csv"))
    clarify = {"plan_id": "x", "goal": "quarterly_profit", "clarification_needed": True,
               "clarification_question": "This file has no cost data. Provide costs, or should I report sales value?",
               "steps": [], "assumptions": []}
    response = build(env, ScriptedQwen(clarify)).answer_question(
        "What was the profit in the latest complete quarter?", csv, "s")
    assert response.status == AnswerStatus.CLARIFICATION_REQUIRED
    assert response.verification is None and "cost" in response.clarification_question


def test_unconfigured_qwen_fails_safely(env, monkeypatch):
    from verisheet import config
    from verisheet.orchestrator import run_agent

    for name in ("QWEN_API_BASE", "QWEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config.get_settings.cache_clear()
    try:
        response = run_agent("total sales?", env["wb"], "s")
    finally:
        config.get_settings.cache_clear()
    assert response.status == AnswerStatus.FAILED and "Qwen endpoint" in response.answer
