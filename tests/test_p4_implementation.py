"""Focused implementation tests for Person 4's analysis, chart, executor and sandbox."""
import json
from pathlib import Path
import pandas as pd
import pytest
from verisheet.tools.analysis import group_aggregate, filter_rows, safe_join, quarter_comparison, run_analysis
from verisheet.tools.chart import make_chart
from verisheet.schemas import ChartSpec, ChartType, AnalysisPlan, AnalysisStep
from verisheet.executor import Executor
from verisheet.sandbox import RestrictedSandbox


def test_group_aggregate_sum_and_filter():
    frame = pd.DataFrame({"Region": ["East", "East", "West"], "Sales": [10, 20, 5]})
    out = group_aggregate(filter_rows(frame, [{"column": "Sales", "operator": "gte", "value": 10}]), ["Region"], {"Sales": "sum"})
    assert out.to_dict("records") == [{"Region": "East", "Sales": 30}]


def test_unknown_operation_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        run_analysis("python", {}, {})


def test_safe_join_rejects_cardinality_violation():
    left = pd.DataFrame({"id": [1, 1], "v": [2, 3]})
    right = pd.DataFrame({"id": [1], "x": [4]})
    spec = {"left_keys": ["id"], "right_keys": ["id"], "expected_cardinality": "many_to_one"}
    with pytest.raises(ValueError, match="Left join keys"):
        safe_join(left, right, spec)


def test_quarter_comparison_last_two_completed_quarters():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-10", "2025-04-10", "2025-07-10", "2025-10-10"]), "profit": [10, 20, 30, 40]})
    out = quarter_comparison(frame, "date", "profit")
    assert out["quarter"].tolist() == ["2025Q2", "2025Q3"]
    assert out["profit"].tolist() == [20.0, 30.0]
    assert out.loc[1, "percent_change"] == 50.0


def test_chart_data_is_loaded_from_result_table(tmp_path):
    table = tmp_path / "result.csv"
    pd.DataFrame({"quarter": ["2025Q2", "2025Q3"], "profit": [20, 30]}).to_csv(table, index=False)
    spec = ChartSpec(chart_type=ChartType.BAR, title="Profit", x_column="quarter", y_column="profit")
    result = make_chart(str(table), spec, str(tmp_path / "charts"))
    saved = json.loads(Path(result["plotted_data_path"]).read_text())
    assert saved == {"quarter": ["2025Q2", "2025Q3"], "profit": [20, 30]}
    assert Path(result["chart_path"]).is_file()


def test_executor_runs_plan_against_csv_fixture(tmp_path):
    csv_path = tmp_path / "sales.csv"
    pd.DataFrame({"Region": ["East", "East", "West"], "Sales": [10, 20, 5]}).to_csv(csv_path, index=False)
    plan = AnalysisPlan(plan_id="p4-test", goal="sales by region", source_sheets=["data"], steps=[
        AnalysisStep(step_id="aggregate", tool="run_analysis", operation="group_aggregate", inputs={"sheet": "data", "group_by": ["Region"], "metrics": {"Sales": "sum"}})
    ])
    results = Executor(str(tmp_path / "artifacts")).execute_plan(plan, str(csv_path), "test-run")
    assert len(results) == 1 and results[0].status.value == "SUCCESS"
    result_frame = pd.read_csv(results[0].result_table_path)
    assert result_frame.to_dict("records") == [{"Region": "East", "Sales": 30}, {"Region": "West", "Sales": 5}]


def test_sandbox_ast_rejects_dynamic_execution_and_network():
    sandbox = RestrictedSandbox()
    with pytest.raises(ValueError): sandbox.validate_script_ast("eval('1 + 1')")
    with pytest.raises(ValueError): sandbox.validate_script_ast("import socket")
    sandbox.validate_script_ast("import math\nvalue = math.sqrt(4)")
