"""Person 5 critic/verifier tests."""

from __future__ import annotations

import json

import pytest

from p5_support import FakeExecutor, FakeQwen, quarter_plan, ranking_plan, write_retail_csv, write_superstore
from verisheet.critic import Critic, CriticMode, extract_json_object, finalize_report, load_prompt
from verisheet.profile import profile_workbook
from verisheet.schemas import AnalysisPlan, AnalysisStep, Decision, VerificationReport
from verisheet.tools import verify as V

ACCEPT = json.dumps({"decision": "ACCEPT", "intent_match": True, "chart_appropriate": True,
                     "rationale": "Plan matches the question."})


@pytest.fixture
def setup(tmp_path):
    workbook = str(write_superstore(tmp_path / "wb.xlsx"))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return workbook, artifacts, profile_workbook(workbook)


def _verify(setup, plan, question, faults=None):
    workbook, artifacts, profile = setup
    results = FakeExecutor(artifacts, faults=[faults or set()]).execute_plan(plan, workbook, "r")
    return results, V.verify_execution(workbook, plan, results, profile, question, artifact_roots=[artifacts])


def test_numeric_verifier_detects_wrong_aggregation(setup):
    plan = quarter_plan()
    results, report = _verify(setup, plan, "last completed quarter profit", {"mean"})
    decision = Critic(mode=CriticMode.DETERMINISTIC).evaluate("q", plan, results, report, setup[2])
    assert decision.decision == Decision.REPAIR
    assert decision.error_category == V.CAT_WRONG_AGGREGATION and decision.failed_step_id == "agg"
    assert decision.evidence["observed"] == "mean(Profit)"


def test_verifier_detects_wrong_quarter(setup):
    plan = quarter_plan()
    results, report = _verify(setup, plan, "last completed quarter profit", {"shift_quarter"})
    decision = Critic(mode=CriticMode.DETERMINISTIC).evaluate("q", plan, results, report, setup[2])
    assert decision.decision == Decision.REPLAN and decision.error_category == V.CAT_WRONG_PERIOD


def test_chart_verifier_detects_plotted_value_mismatch(setup):
    plan = quarter_plan()
    results, report = _verify(setup, plan, "last completed quarter profit", {"alter_plot"})
    decision = Critic(mode=CriticMode.DETERMINISTIC).evaluate("q", plan, results, report, setup[2])
    assert decision.decision == Decision.REPAIR
    assert decision.failed_step_id == "chart" and decision.error_category == V.CAT_CHART_VALUE


def test_critic_never_accepts_without_deterministic_verification(setup):
    plan = quarter_plan()
    # The LLM enthusiastically accepts, but the numbers are wrong: deterministic evidence wins.
    qwen = FakeQwen(ACCEPT, ACCEPT)
    results, report = _verify(setup, plan, "q", {"mean"})
    decision = Critic(client=qwen).evaluate("q", plan, results, report, setup[2])
    assert decision.decision != Decision.ACCEPT and not decision.independent_check_passed
    assert qwen.prompts == []  # the LLM is not even consulted when numbers fail
    # No numeric check could run at all -> never ACCEPT either.
    empty = VerificationReport(verification_id="v", verified=False, numeric_checks_passed=False,
                               chart_checks_passed=True, semantic_checks_passed=True)
    decision = Critic(client=FakeQwen(ACCEPT)).evaluate("q", plan, results, empty, setup[2])
    assert decision.decision == Decision.REPLAN and decision.error_category == V.CAT_NOT_VERIFIABLE
    # LLM-only (E3) may end the loop but never claims independent verification.
    llm_only = Critic(client=FakeQwen(ACCEPT), mode=CriticMode.LLM_ONLY).evaluate("q", plan, results, empty, setup[2])
    assert llm_only.decision == Decision.ACCEPT and not llm_only.independent_check_passed
    assert not finalize_report(empty, llm_only).verified


def test_critic_routes_error_to_correct_transition(setup, tmp_path):
    critic = Critic(mode=CriticMode.DETERMINISTIC)
    # REPAIR: local chart defect.
    plan = quarter_plan()
    results, report = _verify(setup, plan, "q", {"alter_plot"})
    assert critic.evaluate("q", plan, results, report, setup[2]).decision == Decision.REPAIR
    # REPLAN: wrong analytical period in the plan.
    stale = quarter_plan(date_range={"start": "2024-04-01", "end": "2024-06-30"})
    pre = V.verify_plan_semantics("profit in the last completed quarter", stale, setup[2])
    report = VerificationReport(verification_id="v", verified=False, numeric_checks_passed=False,
                                chart_checks_passed=False, semantic_checks_passed=False, findings=pre)
    assert critic.evaluate("q", stale, [], report, setup[2]).decision == Decision.REPLAN
    # CLARIFY: profit requested from a workbook without cost data.
    csv = write_retail_csv(tmp_path / "retail.csv")
    retail_plan = AnalysisPlan(plan_id="r", goal="profit",
                               metric={"name": "profit", "source_columns": ["UnitPrice"], "aggregation": "sum"},
                               steps=[AnalysisStep(step_id="a", tool="run_analysis", operation="group_aggregate")])
    pre = V.verify_plan_semantics("What was the profit last quarter?", retail_plan, profile_workbook(csv))
    report = VerificationReport.model_validate({**report.model_dump(), "findings": pre})
    decision = critic.evaluate("q", retail_plan, [], report, profile_workbook(csv))
    assert decision.decision == Decision.CLARIFY and "cost" in decision.clarification_question
    # Planner-flagged ambiguity is CLARIFY too.
    ambiguous = AnalysisPlan(plan_id="c", goal="x", clarification_needed=True,
                             clarification_question="Which margin definition?")
    assert critic.evaluate("q", ambiguous, [], report, setup[2]).clarification_question == "Which margin definition?"


def test_hybrid_accepts_when_checks_and_semantics_pass(setup):
    plan = quarter_plan()
    results, report = _verify(setup, plan, "profit last completed quarter vs previous")
    qwen = FakeQwen("<think>reasoning</think>```json\n" + ACCEPT + "\n```")
    critic = Critic(client=qwen)
    decision = critic.evaluate("profit last completed quarter vs previous", plan, results, report, setup[2])
    assert decision.decision == Decision.ACCEPT and decision.independent_check_passed
    system, user = qwen.prompts[0]
    assert "VERSION 1.1" in system and "untrusted data" in user and "/no_think" in user
    final = finalize_report(report, decision)
    assert final.verified and final.verification_method == "deterministic_python+qwen_semantic"
    assert critic.last_llm_response.input_tokens == 100


def test_llm_semantic_veto_routes_chart_repair(setup):
    plan = ranking_plan(chart_type="bar")
    results, report = _verify(setup, plan, "top 2 categories by sales")
    veto = json.dumps({"decision": "REPAIR", "intent_match": True, "chart_appropriate": False,
                       "suggested_chart_type": "horizontal_bar", "instruction": "Use a horizontal bar."})
    decision = Critic(client=FakeQwen(veto)).evaluate("top 2 categories", plan, results, report, setup[2])
    assert decision.decision == Decision.REPAIR and decision.failed_step_id == "chart"
    assert decision.evidence["suggested_chart_type"] == "horizontal_bar"
    assert not finalize_report(report, decision).verified


def test_malformed_or_unavailable_llm_keeps_deterministic_verdict(setup):
    plan = quarter_plan()
    results, report = _verify(setup, plan, "q")
    qwen = FakeQwen("not json at all", "still not json")
    critic = Critic(client=qwen)
    decision = critic.evaluate("q", plan, results, report, setup[2])
    assert len(qwen.prompts) == 2  # one bounded retry
    assert decision.decision == Decision.ACCEPT and decision.evidence["semantic_review"] == "unavailable"
    down = Critic(client=FakeQwen(error=RuntimeError("503")))
    assert down.evaluate("q", plan, results, report, setup[2]).decision == Decision.ACCEPT


def test_semantic_review_ignores_injected_instructions_and_bad_fields():
    review = Critic._normalise_review({"decision": "accept", "suggested_chart_type": "pie",
                                       "error_category": "Wrong Metric!; rm -rf", "intent_match": "false"})
    assert review["decision"] == "ACCEPT" and review["suggested_chart_type"] is None
    assert review["error_category"] == "wrongmetricrmrf" and review["intent_match"] is False
    with pytest.raises(ValueError):
        Critic._normalise_review({"decision": "VERIFIED"})


def test_extract_json_object_and_prompt_loading():
    assert extract_json_object('noise {"a": 1} tail') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json_object("<think>{</think> nothing")
    text, version, digest = load_prompt("prompts/critic.txt")
    assert version == "1.1" and len(digest) == 12 and "JSON" in text
    with pytest.raises(FileNotFoundError):
        load_prompt("prompts/does_not_exist.txt")
