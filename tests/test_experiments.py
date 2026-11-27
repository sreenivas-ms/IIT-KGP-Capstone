"""Person 6 experiment runner / scoring / ablation / injection tests (offline; Qwen scripted)."""

import json
from pathlib import Path

import pytest

from test_p5_integration import ScriptedQwen, region_reply
from verisheet.config import get_settings
from verisheet.critic import Critic, CriticMode
from verisheet.executor import Executor
from verisheet.experiments import inject_errors
from verisheet.experiments.ablate import compare_variants
from verisheet.experiments.common import all_gold_matched, load_questions, numbers_in_text, read_jsonl
from verisheet.experiments.gold import GOLD_FUNCTIONS, MINI, compute_gold
from verisheet.experiments.run import VARIANT_LABEL, run_experiment
from verisheet.experiments.score import score_records, score_results
from verisheet.experiments.smoke import build_smoke_set
from verisheet.observability.trace_logger import TraceLogger
from verisheet.orchestrator import AgentOrchestrator
from verisheet.planner import Planner
from verisheet.profile import profile_workbook
from verisheet.schemas import AnswerResponse, AnswerStatus, ExperimentConfig, ExperimentVariant
from verisheet.state import StateStore


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    paths = build_smoke_set(tmp_path / "smoke")
    yield paths
    get_settings.cache_clear()


def config(variant: str, questions: str) -> ExperimentConfig:
    return ExperimentConfig(system=ExperimentVariant(variant), split="dev", question_file=questions,
                            model_name="Qwen/Qwen3-8B", prompt_version="test")


class FakeSystem:
    """Stands in for a variant: never calls Qwen; answers from the gold to test the plumbing."""

    def __init__(self, gold):
        self.gold = gold

    def prompt_versions(self):
        return {"planner": "test"}

    def answer(self, question, workbook, session_id):
        return AnswerResponse(status=AnswerStatus.NOT_VERIFIED, answer="fake", trace_id="none"), {}


def test_all_variants_use_identical_test_question_ids(smoke, tmp_path):
    out = tmp_path / "runs"
    for variant in ExperimentVariant:
        run_experiment(config(variant.value, smoke["questions"]), out / variant.value,
                       system_factory=lambda c: FakeSystem(None))
    metas = [json.loads((out / v.value / "run_metadata.json").read_text()) for v in ExperimentVariant]
    assert len({tuple(m["question_ids"]) for m in metas}) == 1 and len(metas[0]["question_ids"]) == 12
    assert len({m["test_set_sha256"] for m in metas}) == 1
    assert {m["variant_label"] for m in metas} == {"E1", "E2", "E3", "E4", "E5"}
    result = compare_variants([str(out / v.value) for v in ExperimentVariant], str(tmp_path / "abl.json"),
                              smoke["gold"])
    assert [row["label"] for row in result["table"]] == ["E1", "E2", "E3", "E4", "E5"]
    assert (tmp_path / "abl.md").read_text().startswith("| Variant |")

    # a variant run on different questions must be refused
    run_experiment(config("full", smoke["questions"]), out / "short", system_factory=lambda c: FakeSystem(None),
                   limit=3)
    with pytest.raises(ValueError, match="identical question ids"):
        compare_variants([str(out / "direct"), str(out / "short")], str(tmp_path / "x.json"), smoke["gold"])


def test_question_file_is_validated_and_missing_file_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="Person 1"):
        load_questions(tmp_path / "missing.jsonl")
    bad = tmp_path / "q.jsonl"
    bad.write_text(json.dumps({"id": "x", "question": "q"}) + "\n")
    with pytest.raises(ValueError, match="question_task schema"):
        load_questions(bad)


def _record(qid, status, numbers=(), plotted=(), chart=None, repairs=0, **kw):
    return {"question_id": qid, "status": status, "table_numbers": list(numbers), "answer_numbers": [],
            "plotted_numbers": list(plotted), "chart_type": chart, "repairs": repairs, "replans": 0,
            "latency_ms": kw.get("latency", 100.0), "input_tokens": 10, "output_tokens": 5, "llm_calls": 2,
            "actions": 3, "error": None}


def test_scorer_reports_numeric_plot_repair_and_false_verification_metrics():
    gold = {
        "a": {"id": "a", "gold_value": {"East": 50.0, "West": 90.0}, "must_refuse": False, "accepted_charts": ["bar"]},
        "b": {"id": "b", "gold_value": 140.0, "must_refuse": False, "accepted_charts": ["table"]},
        "c": {"id": "c", "gold_value": 10.0, "must_refuse": False, "accepted_charts": ["bar"]},
        "d": {"id": "d", "gold_value": None, "must_refuse": True, "accepted_charts": []},
    }
    records = [
        _record("a", "VERIFIED", [50, 90], plotted=[50, 90], chart="bar", repairs=1),   # correct, repaired
        _record("b", "VERIFIED", [139]),                                               # VERIFIED but wrong
        _record("c", "NOT_VERIFIED", [10], plotted=[11], chart="line", latency=300.0),  # right number, bad plot
        _record("d", "CLARIFICATION_REQUIRED"),                                        # correct refusal
    ]
    m = score_records(records, gold)
    assert m["numeric_accuracy"] == round(2 / 3, 4)
    assert m["task_success"] == 0.75
    assert m["false_verification_rate"] == 0.5 and m["n_verified"] == 2
    assert m["verified_coverage"] == round(2 / 3, 4) and m["verified_correct_coverage"] == round(1 / 3, 4)
    assert m["plot_data_accuracy"] == 0.5 and m["chart_appropriateness"] == 0.5
    assert m["refusal_accuracy"] == 1.0 and m["repair_success"] == 1.0
    assert m["execution_success"] == 1.0 and m["mean_actions"] == 3 and m["latency_ms_p95"] == 300.0


def test_number_matching_rules():
    assert numbers_in_text("Profit was 1,250.50 (up 20%) vs (300)") == [1250.5, 20.0, 0.2, -300.0]
    assert all_gold_matched({"East": 0.04}, [0.04001]) and not all_gold_matched(140.0, [139.0])
    assert not all_gold_matched(None, [1.0])


def test_gold_is_computed_not_hand_entered():
    assert compute_gold("sales_by_region", MINI) == {"East": 50.0, "West": 90.0}
    assert compute_gold(None, MINI) is None and len(GOLD_FUNCTIONS) >= 10


def _scripted_orchestrator(tmp_path, *replies):
    return AgentOrchestrator(
        profiler=profile_workbook, retriever=None, planner=Planner(client=ScriptedQwen(*replies)),
        executor=Executor(artifact_dir=str(tmp_path / "artifacts")), critic=Critic(mode=CriticMode.DETERMINISTIC),
        state_store=StateStore(str(tmp_path / "sessions")), trace_logger=TraceLogger(str(tmp_path / "traces")),
        artifact_roots=[tmp_path / "artifacts"])


def test_injection_suite_runs_twenty_cases(tmp_path, monkeypatch):
    cases = inject_errors.load_cases(inject_errors.DEFAULT_CASES)
    assert len(cases) >= 20
    required = {"wrong_aggregation", "wrong_period", "duplicated_join", "missing_metric", "missing_sheet",
                "invalid_chart_arrays", "prompt_injection", "unsafe_code"}
    assert required <= {c["category"] for c in cases}
    books = inject_errors.build_workbooks(tmp_path / "books")
    assert {c["workbook"] for c in cases} <= set(books)

    # Real pipeline, scripted Qwen: wrong aggregation and tampered chart must fire and be caught.
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    get_settings.cache_clear()
    orchestrator = _scripted_orchestrator(tmp_path, *[region_reply()] * 6)
    records = inject_errors.run_injection_suite("test", str(inject_errors.DEFAULT_CASES), only=["c01", "c10"],
                                                out_dir=tmp_path / "inj", orchestrator=orchestrator)
    get_settings.cache_clear()
    assert [r["case_id"] for r in records] == ["c01", "c10"]
    assert all(r["fault_fired"] == 1 and r["passed"] and not r["verified_wrong"] for r in records)
    assert all(r["repaired"] for r in records)            # detected, repaired, verified correct
    summary = json.loads((tmp_path / "inj" / "injection_metadata.json").read_text())["summary"]
    assert summary["error_detection_recall"] == 1.0 and summary["untriggered_faults"] == []


def test_untriggered_fault_is_not_counted_as_detected():
    case = {"id": "x", "expect": "detect", "fault": "agg_swap:mean"}
    record = {"status": "NOT_VERIFIED", "table_numbers": [], "plotted_numbers": [], "answer_numbers": [],
              "answer": "", "fault_fired": 0, "repairs": 0, "replans": 0}
    verdict = inject_errors.judge_case(case, record, 140.0)
    assert verdict["passed"] is False and verdict["triggered"] is False


def test_experiment_trace_records_model_and_prompt_versions(smoke, tmp_path):
    orchestrator = _scripted_orchestrator(tmp_path, region_reply(), region_reply())

    class Scripted:
        def prompt_versions(self):
            return {"planner": orchestrator.planner.prompt_version}

        def answer(self, question, workbook, session_id):
            return orchestrator.answer_question(question, workbook, session_id), {}

    records = run_experiment(config("full", smoke["questions"]), tmp_path / "run",
                             system_factory=lambda c: Scripted(), limit=2)
    meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    assert meta["model_name"] == "Qwen/Qwen3-8B" and meta["prompt_versions"]["planner"]
    assert meta["test_set_sha256"] and meta["git_commit"] is not None and meta["temperature"] == 0.0
    events = TraceLogger(str(tmp_path / "traces")).load_trace(records[1]["trace_id"])
    config_event = next(e for e in events if e.event_type == "experiment_config")
    assert config_event.metadata["variant"] == "full" and config_event.metadata["model_name"] == "Qwen/Qwen3-8B"
    assert config_event.metadata["prompt_versions"]["planner"] and config_event.metadata["question_id"] == "s02"
    assert records[1]["status"] == "VERIFIED" and records[1]["input_tokens"] > 0
    summary = score_results(str(tmp_path / "run"), smoke["gold"], str(tmp_path / "score.json"))
    assert summary["variants"]["full"]["prompt_versions"]["planner"] and summary["gold_sha256"]
    assert VARIANT_LABEL[ExperimentVariant.FULL] == "E5" and read_jsonl(tmp_path / "run" / "results.jsonl")


def test_computed_but_withheld_numbers_count_as_computed_not_delivered():
    gold = {"a": {"id": "a", "gold_value": 140.0, "must_refuse": False, "accepted_charts": []}}
    withheld = {**_record("a", "NOT_VERIFIED"), "executed_numbers": [140.0]}     # E2: computed, not shown
    m = score_records([withheld], gold)
    assert m["numeric_accuracy"] == 1.0 and m["delivered_numeric_accuracy"] == 0.0
