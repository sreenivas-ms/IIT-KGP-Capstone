"""Person 5 full loop and state management tests."""

from __future__ import annotations

import json

import pytest

from p5_support import (
    FakeExecutor, FakeQwen, ListTraceLogger, ScriptedPlanner, quarter_plan, ranking_plan,
    write_retail_csv, write_superstore,
)
from verisheet.critic import Critic, CriticMode
from verisheet.orchestrator import AgentOrchestrator, compose_answer, run_agent
from verisheet.profile import profile_workbook
from verisheet.schemas import (
    AgentState, AnalysisPlan, AnalysisStep, AnswerResponse, AnswerStatus, ChartType, MetricSpec,
    VerificationReport,
)
from verisheet.state import StateStore

Q1 = "What was recorded profit in the last fully completed quarter, compared with the previous quarter?"


@pytest.fixture
def env(tmp_path):
    workbook = str(write_superstore(tmp_path / "superstore.xlsx"))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return {"wb": workbook, "artifacts": artifacts, "tmp": tmp_path,
            "store": StateStore(str(tmp_path / "sessions")), "trace": ListTraceLogger()}


def make(env, planner, executor, critic=None, **kwargs) -> AgentOrchestrator:
    return AgentOrchestrator(
        profiler=profile_workbook, retriever=None, planner=planner, executor=executor,
        critic=critic if critic is not None else Critic(mode=CriticMode.DETERMINISTIC),
        state_store=env["store"], trace_logger=env["trace"], artifact_roots=[env["artifacts"]],
        trace_dir=str(env["tmp"] / "traces"), **kwargs)


def _types(orch):
    return [(e.actor, e.event_type, e.status) for e in orch.last_events]


def test_orchestrator_returns_verified_answer_when_all_checks_pass(env):
    qwen = FakeQwen(json.dumps({"decision": "ACCEPT", "intent_match": True, "chart_appropriate": True}))
    orch = make(env, ScriptedPlanner(quarter_plan()), FakeExecutor(env["artifacts"]), Critic(client=qwen))
    response = orch.answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.VERIFIED and response.verification.verified
    assert "2024Q3" in response.answer and "130.00" in response.answer and "40.00" in response.answer
    assert response.chart_spec.chart_type == ChartType.BAR and response.result_table_path
    assert response.workbook_id.startswith("sha256:")
    # One trace ID joins every event; planner -> executor -> verifier -> critic -> final.
    events = env["trace"].events
    assert {e.trace_id for e in events} == {response.trace_id}
    actors = [e.actor for e in events]
    assert actors.index("planner") < actors.index("executor") < actors.index("critic")
    assert events[-1].event_type == "final_response" and events[-1].metadata["actions"] == 4
    critic_event = next(e for e in events if e.actor == "critic")
    assert critic_event.input_tokens == 100 and critic_event.prompt_version == "1.1"
    # Verified context is persisted for follow-ups.
    state = env["store"].load("s1", response.workbook_id)
    assert state.accepted_period["quarter"] == "2024Q3" and state.accepted_metric.name == "profit"


def test_orchestrator_repairs_failed_step_at_most_twice(env):
    # Chart value is corrupted on every run: two targeted repairs, then NOT_VERIFIED.
    executor = FakeExecutor(env["artifacts"], always={"alter_plot"})
    orch = make(env, ScriptedPlanner(quarter_plan()), executor)
    response = orch.answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.NOT_VERIFIED and "repair budget" in response.answer
    assert executor.calls == [("plan", ["agg", "chart"]), ("step", ["chart"]), ("step", ["chart"])]
    assert [e.status for e in orch.last_events if e.event_type == "transition"].count("REPAIR") == 2
    assert orch.last_events[-1].metadata["actions"] <= 10


def test_orchestrator_targeted_repair_succeeds(env):
    executor = FakeExecutor(env["artifacts"], faults=[{"alter_plot"}])
    response = make(env, ScriptedPlanner(quarter_plan()), executor).answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.VERIFIED
    assert executor.calls == [("plan", ["agg", "chart"]), ("step", ["chart"])]


def test_wrong_aggregation_repairs_step_and_dependants(env):
    executor = FakeExecutor(env["artifacts"], faults=[{"mean"}])
    response = make(env, ScriptedPlanner(quarter_plan()), executor).answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.VERIFIED
    assert executor.calls[1:] == [("step", ["agg"]), ("step", ["chart"])]


def test_orchestrator_replans_for_wrong_analytical_approach(env):
    wrong = quarter_plan("plan-wrong", date_range={"start": "2024-04-01", "end": "2024-06-30"})
    planner = ScriptedPlanner(wrong, quarter_plan("plan-right"))
    executor = FakeExecutor(env["artifacts"])
    response = make(env, planner, executor).answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.VERIFIED
    feedback = planner.revise_calls[0]["feedback"]
    assert feedback["decision"] == "REPLAN" and feedback["error_category"] == "wrong_period"
    assert feedback["evidence"]["expected"] == "2024Q3"
    assert executor.calls == [("plan", ["agg", "chart"])]  # the wrong plan was never executed


def test_executor_wrong_period_triggers_replan(env):
    planner = ScriptedPlanner(quarter_plan("p1"), quarter_plan("p2"))
    executor = FakeExecutor(env["artifacts"], faults=[{"shift_quarter"}])
    response = make(env, planner, executor).answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.VERIFIED and len(planner.revise_calls) == 1


def test_orchestrator_clarifies_when_data_is_insufficient(env):
    csv = str(write_retail_csv(env["tmp"] / "retail.csv"))
    plan = AnalysisPlan(plan_id="r", goal="profit",
                        metric={"name": "profit", "source_columns": ["UnitPrice", "Quantity"], "aggregation": "sum"},
                        steps=[AnalysisStep(step_id="a", tool="run_analysis", operation="time_aggregate",
                                            inputs={"date_column": "InvoiceDate", "metric_column": "UnitPrice",
                                                    "frequency": "Q"})])
    executor = FakeExecutor(env["artifacts"])
    orch = make(env, ScriptedPlanner(plan), executor)
    response = orch.answer_question("What was the profit in the latest complete quarter?", csv, "s2")
    assert response.status == AnswerStatus.CLARIFICATION_REQUIRED
    assert "cost" in response.clarification_question and executor.calls == []
    # Planner-originated clarification also stops before execution.
    ask = AnalysisPlan(plan_id="c", goal="x", clarification_needed=True, clarification_question="Which currency?")
    response = make(env, ScriptedPlanner(ask), executor).answer_question("total in USD", env["wb"], "s3")
    assert response.clarification_question == "Which currency?" and executor.calls == []


def test_state_store_does_not_persist_unverified_answer_as_fact(env):
    executor = FakeExecutor(env["artifacts"], always={"alter_plot"})
    response = make(env, ScriptedPlanner(quarter_plan()), executor).answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.NOT_VERIFIED
    state = env["store"].load("s1", response.workbook_id)
    assert state.accepted_metric is None and state.accepted_period is None and state.revision == 0
    assert state.history[-1].content_summary.startswith("[not verified")
    # Direct API check: a VERIFIED status is required to update accepted facts.
    fake = AnswerResponse(status=AnswerStatus.NOT_VERIFIED, answer="profit was 999", trace_id="t")
    updated = env["store"].update_from_verified_response(AgentState(session_id="x"), fake, quarter_plan())
    assert updated.accepted_metric is None and "999" not in updated.history[-1].content_summary


def test_follow_up_receives_verified_context_and_workbook_change_isolates_it(env):
    planner = ScriptedPlanner(quarter_plan("p1"), ranking_plan("p2"), ranking_plan("p3"))
    orch = make(env, planner, FakeExecutor(env["artifacts"]))
    first = orch.answer_question(Q1, env["wb"], "s1")
    assert first.status == AnswerStatus.VERIFIED
    orch.answer_question("Which two categories sold the most?", env["wb"], "s1")
    previous = planner.generate_calls[1]["previous_state"]
    assert previous.accepted_period["quarter"] == "2024Q3"
    assert previous.history[0].role == "user"
    other = str(write_superstore(env["tmp"] / "other.xlsx", data_through="2024-12-31"))
    orch.answer_question("Which two categories sold the most?", other, "s1")
    assert planner.generate_calls[2]["previous_state"].accepted_period is None


def test_action_budget_cannot_be_exceeded(env):
    executor = FakeExecutor(env["artifacts"], always={"alter_plot"})
    orch = make(env, ScriptedPlanner(quarter_plan()), executor, max_actions=5, max_repairs=5)
    response = orch.answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.NOT_VERIFIED and "budget" in response.answer
    assert orch.last_events[-1].metadata["actions"] <= 5
    assert any(e.event_type == "budget_exceeded" for e in orch.last_events)


def test_pre_execution_chart_type_repair(env):
    plan = ranking_plan(chart_type="line")
    executor = FakeExecutor(env["artifacts"])
    response = make(env, ScriptedPlanner(plan), executor).answer_question(
        "Which two categories have the highest sales?", env["wb"], "s1")
    assert response.status == AnswerStatus.VERIFIED
    assert response.chart_spec.chart_type == ChartType.HORIZONTAL_BAR


def test_executor_exception_and_planner_failures_are_safe(env):
    response = make(env, ScriptedPlanner(quarter_plan()), FakeExecutor(env["artifacts"], always={"raise"})) \
        .answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.NOT_VERIFIED
    bad = make(env, ScriptedPlanner(error=ValueError("malformed JSON")), FakeExecutor(env["artifacts"]))
    response = bad.answer_question(Q1, env["wb"], "s1")
    assert response.status == AnswerStatus.FAILED and "planner" in response.answer
    stub = make(env, ScriptedPlanner(error=NotImplementedError("P3")), FakeExecutor(env["artifacts"]))
    assert stub.answer_question(Q1, env["wb"], "s1").status == AnswerStatus.FAILED


def test_invalid_inputs_fail_safely(env):
    orch = make(env, ScriptedPlanner(quarter_plan()), FakeExecutor(env["artifacts"]))
    assert orch.answer_question(Q1, env["wb"], "../../etc/passwd").status == AnswerStatus.FAILED
    assert orch.answer_question("  ", env["wb"], "s1").status == AnswerStatus.FAILED
    missing = orch.answer_question(Q1, str(env["tmp"] / "nope.xlsx"), "s1")
    assert missing.status == AnswerStatus.FAILED and "Traceback" not in missing.answer


def test_trace_falls_back_to_jsonl_when_logger_is_a_stub(env):
    class StubLogger:
        def log_event(self, event):
            raise NotImplementedError

    orch = make(env, ScriptedPlanner(quarter_plan()), FakeExecutor(env["artifacts"]))
    orch.trace_logger = StubLogger()
    response = orch.answer_question(Q1, env["wb"], "s1")
    lines = (env["tmp"] / "traces" / f"{response.trace_id}.jsonl").read_text().splitlines()
    assert len(lines) == len(orch.last_events) and json.loads(lines[-1])["event_type"] == "final_response"


def test_no_critic_variant_is_never_verified(env):
    orch = make(env, ScriptedPlanner(quarter_plan()), FakeExecutor(env["artifacts"]))
    orch.critic = None
    assert orch.answer_question(Q1, env["wb"], "s1").status == AnswerStatus.NOT_VERIFIED


def test_run_agent_uses_injected_orchestrator_and_handles_stubs(env):
    orch = make(env, ScriptedPlanner(quarter_plan()), FakeExecutor(env["artifacts"]))
    assert run_agent(Q1, env["wb"], "s1", orchestrator=orch).status == AnswerStatus.VERIFIED
    default = run_agent(Q1, env["wb"], "s1")  # teammates' stubs not implemented yet
    assert default.status == AnswerStatus.FAILED and default.trace_id.startswith("trace-")


def test_compose_answer_reports_undefined_growth_on_zero_baseline():
    plan = quarter_plan()
    report = VerificationReport(
        verification_id="v", verified=True, numeric_checks_passed=True, chart_checks_passed=True,
        semantic_checks_passed=True,
        recomputed_values={"agg": {"scalars": {"quarter": "2024Q3", "previous_quarter": "2024Q2", "current": 5.0,
                                               "previous": 0.0, "absolute_change": 5.0, "pct_change": None}}})
    assert "undefined" in compose_answer(plan, report)


def test_state_store_atomic_isolated_and_corruption_safe(tmp_path):
    store = StateStore(str(tmp_path / "s"))
    state = AgentState(session_id="abc", workbook_id="sha256:1", accepted_metric=MetricSpec(
        name="profit", source_columns=["Profit"], aggregation="sum"))
    store.save(state)
    assert store.load("abc", "sha256:1").accepted_metric.name == "profit"
    assert store.load("abc", "sha256:2").accepted_metric is None
    files = list((tmp_path / "s").glob("state_*.json"))
    assert len(files) == 1 and "abc" not in files[0].name and not list((tmp_path / "s").glob("*.tmp"))
    files[0].write_text("{corrupt")
    assert store.load("abc", "sha256:1").accepted_metric is None
    assert list((tmp_path / "s").glob("*.corrupt"))
    with pytest.raises(ValueError):
        store.load("../x")
    for i in range(30):
        state = StateStore.append_history(state, "user", "q" * 1000 + str(i))
    assert len(state.history) == 20 and len(state.history[-1].content_summary) <= 400
