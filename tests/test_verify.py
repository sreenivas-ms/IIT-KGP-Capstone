"""Person 5: deterministic verifier tests (numeric recomputation, chart fidelity, plan checks)."""

from __future__ import annotations

import pandas as pd
import pytest

from p5_support import FakeExecutor, quarter_plan, ranking_plan, write_retail_csv, write_superstore
from verisheet.profile import profile_workbook
from verisheet.schemas import (
    AnalysisPlan, AnalysisStep, ChartSpec, ExecutionResult, JoinSpec, RunStatus,
)
from verisheet.tools import verify as V


@pytest.fixture
def workbook(tmp_path):
    return str(write_superstore(tmp_path / "superstore.xlsx"))


@pytest.fixture
def artifacts(tmp_path):
    path = tmp_path / "artifacts"
    path.mkdir()
    return path


def _run(plan, workbook, artifacts, faults=None):
    return FakeExecutor(artifacts, faults=[faults or set()]).execute_plan(plan, workbook, "run-1")


def _failed(findings, category):
    return [f for f in findings if not f["passed"] and f["category"] == category]


def test_canonical_quarter_comparison_matches_hand_computed_values(workbook, artifacts):
    plan = quarter_plan()
    results = _run(plan, workbook, artifacts)
    report = V.verify_execution(workbook, plan, results, profile_workbook(workbook),
                                "What was profit in the last completed quarter vs the previous one?",
                                artifact_roots=[artifacts])
    assert report.verified, [f for f in report.findings if not f.passed]
    scalars = report.recomputed_values["agg"]["scalars"]
    assert scalars["quarter"] == "2024Q3" and scalars["previous_quarter"] == "2024Q2"
    assert scalars["current"] == pytest.approx(130.0)
    assert scalars["previous"] == pytest.approx(40.0)
    assert scalars["pct_change"] == pytest.approx(225.0)
    assert report.recomputed_values["period"]["quarter"] == "2024Q3"


def test_numeric_verifier_detects_wrong_aggregation(workbook, artifacts):
    plan = quarter_plan()
    results = _run(plan, workbook, artifacts, {"mean"})
    report = V.recompute_check(workbook, plan, results[0], artifact_roots=[artifacts])
    assert not report.verified and not report.numeric_checks_passed
    bad = [f for f in report.findings if f.category == V.CAT_WRONG_AGGREGATION]
    assert bad and bad[0].observed == "mean(Profit)"


def test_verifier_detects_wrong_quarter(workbook, artifacts):
    plan = quarter_plan()
    results = _run(plan, workbook, artifacts, {"shift_quarter"})
    report = V.recompute_check(workbook, plan, results[0], artifact_roots=[artifacts])
    assert not report.verified
    assert any(f.category == V.CAT_WRONG_PERIOD for f in report.findings if not f.passed)


def test_data_through_metadata_overrides_observed_date(tmp_path):
    path = write_superstore(tmp_path / "wb.xlsx", data_through="2024-12-31")
    frames = V.load_source_frames(path)
    quarter, note = V.completed_quarter(frames["Orders"]["Order_Date"], V.find_data_through(frames))
    assert str(quarter) == "2024Q4" and "Data_Through" in note
    mid = V.completed_quarter(frames["Orders"]["Order_Date"], pd.Timestamp("2024-11-15"))[0]
    assert str(mid) == "2024Q3"


def test_chart_verifier_detects_plotted_value_mismatch(workbook, artifacts):
    plan = quarter_plan()
    results = _run(plan, workbook, artifacts, {"alter_plot"})
    report = V.verify_execution(workbook, plan, results, None, "", artifact_roots=[artifacts])
    assert report.numeric_checks_passed and not report.chart_checks_passed and not report.verified
    assert any(f.category == V.CAT_CHART_VALUE and not f.passed for f in report.findings)


def test_chart_verifier_detects_missing_category(workbook, artifacts):
    plan = ranking_plan(n=3)
    results = _run(plan, workbook, artifacts, {"drop_category"})
    report = V.verify_execution(workbook, plan, results, None, "Top 3 categories by sales",
                                artifact_roots=[artifacts])
    assert any(f.category == V.CAT_CHART_CATEGORY and not f.passed for f in report.findings)


def test_ranking_plotted_as_line_is_flagged(workbook, artifacts):
    plan = ranking_plan(chart_type="line")
    results = _run(plan, workbook, artifacts)
    findings = V.verify_chart_data(results[1], result_table=pd.read_csv(results[0].result_table_path),
                                   question="Which two categories have the most sales?")
    flagged = _failed(findings, V.CAT_CHART_TYPE)
    assert flagged and flagged[0]["expected"] == "horizontal_bar"


def test_bar_chart_with_truncated_axis_is_flagged(workbook, artifacts):
    plan = quarter_plan()
    results = _run(plan, workbook, artifacts)
    chart = results[1].model_copy(update={"chart_spec": plan.chart_spec.model_copy(update={"baseline": 25.0})})
    findings = V.verify_chart_data(chart, result_table=pd.read_csv(results[0].result_table_path), key_kind="Q")
    assert _failed(findings, V.CAT_AXIS_TRUNCATION)


def test_top_n_and_wrong_order(workbook, artifacts):
    plan = ranking_plan(n=2)
    results = _run(plan, workbook, artifacts)
    report = V.verify_execution(workbook, plan, results, None, "top 2 categories", artifact_roots=[artifacts])
    assert report.verified
    rows = report.recomputed_values["rank"]["rows"]
    assert [r["Category"] for r in rows] == ["B", "C"]  # B=1300, C=1300 -> tie broken by key; A=1000
    # Reverse the executor table: same values, wrong ranking order.
    path = results[0].result_table_path
    pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    report = V.recompute_check(workbook, plan, results[0], artifact_roots=[artifacts])
    assert any(f.category == V.CAT_CHART_ORDER and not f.passed for f in report.findings)


def _join_plan(cardinality):
    spec = {"left_sheet": "Orders", "right_sheet": "Returns", "left_keys": ["Order_ID"],
            "right_keys": ["Order_ID"], "expected_cardinality": cardinality}
    return AnalysisPlan(
        plan_id="join", goal="orders_with_returns", source_sheets=["Orders", "Returns"],
        joins=[JoinSpec(**spec)],
        steps=[AnalysisStep(step_id="j", tool="run_analysis", operation="safe_join", inputs={"join_spec": spec})])


def test_join_multiplication_is_detected(workbook, artifacts):
    # Returns has Order_ID O2 twice: a many_to_one join would multiply order measures.
    plan = _join_plan("many_to_one")
    result = ExecutionResult(run_id="r", step_id="j", status=RunStatus.SUCCESS, row_count_out=9)
    report = V.recompute_check(workbook, plan, result, artifact_roots=[artifacts])
    bad = [f for f in report.findings if not f.passed]
    assert bad and bad[0].category == V.CAT_JOIN_MULTIPLICATION and not report.verified
    # Declared one_to_many with unique left keys is a legitimate 9-row join.
    honest = _join_plan("one_to_many")
    assert V.recompute_check(workbook, honest, result, artifact_roots=[artifacts]).verified
    inflated = result.model_copy(update={"row_count_out": 12})
    assert not V.recompute_check(workbook, honest, inflated, artifact_roots=[artifacts]).verified
    # The pre-execution check flags the non-unique right key before anything runs.
    findings = V.verify_plan_semantics("orders with returns", plan, profile_workbook(workbook))
    assert _failed(findings, V.CAT_JOIN_GRAIN)


def test_tool_errors_route_by_retryability():
    from verisheet.schemas import ToolError
    assert V.tool_error_category(ToolError(code="E", category="TimeoutError", message="t", retryable=True)) \
        == V.CAT_EXECUTION_FAILED
    assert V.tool_error_category(ToolError(code="E", category="ValueError", message="Unknown column(s): ['X']")) \
        == V.CAT_UNKNOWN_REFERENCE
    assert V.tool_error_category(ToolError(code="E", category="ValueError", message="n must be positive")) \
        == V.CAT_TOOL_REJECTED


def test_missing_profit_metric_routes_to_clarify(tmp_path):
    csv = write_retail_csv(tmp_path / "retail.csv")
    plan = AnalysisPlan(
        plan_id="p", goal="quarterly_profit",
        metric={"name": "profit", "source_columns": ["UnitPrice", "Quantity"], "aggregation": "sum"},
        steps=[AnalysisStep(step_id="a", tool="run_analysis", operation="time_aggregate",
                            inputs={"date_column": "InvoiceDate", "metric_column": "UnitPrice", "frequency": "Q"})])
    findings = V.verify_plan_semantics("What was quarterly profit?", plan, profile_workbook(csv))
    assert findings[0]["category"] == V.CAT_METRIC_UNAVAILABLE
    assert V.ROUTING[findings[0]["category"]] == "CLARIFY"


def test_year_outside_data_routes_to_clarify(tmp_path):
    csv = write_retail_csv(tmp_path / "retail.csv")
    plan = AnalysisPlan(
        plan_id="p", goal="trend",
        metric={"name": "sales_value", "source_columns": ["UnitPrice"], "aggregation": "sum"},
        steps=[AnalysisStep(step_id="a", tool="run_analysis", operation="time_aggregate",
                            inputs={"date_column": "InvoiceDate", "metric_column": "UnitPrice", "frequency": "M"})])
    findings = V.verify_plan_semantics("Show sales trend for 2024.", plan, profile_workbook(csv))
    assert _failed(findings, V.CAT_PERIOD_UNAVAILABLE)


def test_plan_with_wrong_explicit_quarter_is_flagged(workbook):
    plan = quarter_plan(quarter="2024Q3")
    findings = V.verify_plan_semantics("Did profit in Q2 2024 increase?", plan, profile_workbook(workbook))
    assert _failed(findings, V.CAT_WRONG_PERIOD)
    stale = quarter_plan(date_range={"start": "2024-04-01", "end": "2024-06-30"})
    findings = V.verify_plan_semantics("Profit in the last completed quarter?", stale, profile_workbook(workbook))
    assert _failed(findings, V.CAT_WRONG_PERIOD)[0]["expected"] == "2024Q3"


def test_margin_must_be_ratio_of_sums(workbook):
    plan = AnalysisPlan(
        plan_id="m", goal="margin",
        metric={"name": "profit_margin", "source_columns": ["Profit", "Sales"], "aggregation": "mean",
                "formula": "mean(Profit/Sales)"},
        group_by=["Region"],
        steps=[AnalysisStep(step_id="a", tool="run_analysis", operation="group_aggregate",
                            inputs={"group_by": ["Region"], "metric_column": "Profit"})])
    findings = V.verify_plan_semantics("Which region had the highest profit margin?", plan, profile_workbook(workbook))
    assert _failed(findings, V.CAT_WRONG_MARGIN)


def test_grouped_ratio_recomputes_margin_as_ratio_of_sums(workbook, artifacts, tmp_path):
    plan = AnalysisPlan(
        plan_id="m2", goal="margin",
        metric={"name": "margin", "source_columns": ["Profit", "Sales"], "aggregation": "ratio_of_sums",
                "formula": "sum(Profit)/sum(Sales)"},
        steps=[AnalysisStep(step_id="a", tool="run_analysis", operation="grouped_ratio",
                            inputs={"sheet": "Orders", "group_by": ["Region"], "numerator": "Profit",
                                    "denominator": "Sales", "output": "margin"})])
    # East: profit 10+30+50+70=160, sales 1600 -> 0.1 ; West: 20-40+60+80=120, sales 2000 -> 0.06
    table = artifacts / "m.csv"
    pd.DataFrame({"Region": ["East", "West"], "margin": [0.1, 0.06]}).to_csv(table, index=False)
    result = ExecutionResult(run_id="r", step_id="a", status=RunStatus.SUCCESS, result_table_path=str(table))
    assert V.recompute_check(workbook, plan, result, artifact_roots=[artifacts]).verified
    # Mean of row margins would differ -> flagged.
    pd.DataFrame({"Region": ["East", "West"], "margin": [0.1, 0.0458]}).to_csv(table, index=False)
    assert not V.recompute_check(workbook, plan, result, artifact_roots=[artifacts]).verified


def test_unsupported_operation_is_never_verified(workbook, artifacts):
    plan = quarter_plan()
    plan.steps[0] = plan.steps[0].model_copy(update={"operation": "magic_forecast"})
    result = ExecutionResult(run_id="r", step_id="agg", status=RunStatus.SUCCESS)
    report = V.recompute_check(workbook, plan, result, artifact_roots=[artifacts])
    assert not report.verified and report.findings[0].category == V.CAT_NOT_VERIFIABLE


def test_result_table_outside_artifact_root_is_rejected(workbook, artifacts, tmp_path):
    plan = quarter_plan()
    results = _run(plan, workbook, artifacts)
    outside = tmp_path / "outside.csv"
    pd.read_csv(results[0].result_table_path).to_csv(outside, index=False)
    escaped = results[0].model_copy(update={"result_table_path": str(artifacts / ".." / "outside.csv")})
    report = V.recompute_check(workbook, plan, escaped, artifact_roots=[artifacts])
    assert any(f.category == V.CAT_UNTRUSTED_ARTIFACT for f in report.findings)
    assert not report.verified


def test_failed_execution_is_not_verified(workbook, artifacts):
    plan = quarter_plan()
    results = FakeExecutor(artifacts, faults=[{"fail_chart"}]).execute_plan(plan, workbook, "r")
    report = V.verify_execution(workbook, plan, results, None, "", artifact_roots=[artifacts])
    assert not report.verified and not report.chart_checks_passed


def test_plan_structure_and_code_injection_checks(workbook):
    plan = AnalysisPlan(
        plan_id="bad", goal="x",
        metric={"name": "sales", "source_columns": ["Sales"], "aggregation": "sum"},
        steps=[AnalysisStep(step_id="a", tool="run_analysis", operation="group_aggregate",
                            inputs={"group_by": ["__import__('os').system('id')"]}, depends_on=["b"]),
               AnalysisStep(step_id="b", tool="run_analysis", operation="group_aggregate", depends_on=["a"])])
    findings = V.verify_plan_semantics("total sales", plan, profile_workbook(workbook))
    categories = {f["category"] for f in findings if not f["passed"]}
    assert {V.CAT_INVALID_PLAN, V.CAT_UNSAFE_PLAN, V.CAT_UNKNOWN_REFERENCE} <= categories


def test_quarter_label_formats_are_normalised():
    assert V._norm_key("Q3 2024", "Q") == V._norm_key("2024-Q3", "Q") == V._norm_key("2024Q3", "Q") == "2024Q3"
    assert V._norm_key("2024-03-31", "M") == "2024-03"


def test_filters_fail_closed_on_unknown_operator(workbook):
    frames = V.load_source_frames(workbook)
    with pytest.raises(ValueError):
        V._apply_filters(frames["Orders"], [{"column": "Region", "op": "regex", "value": ".*"}])
    out = V._apply_filters(frames["Orders"], [{"column": "Region", "op": "==", "value": "East"}])
    assert len(out) == 4
