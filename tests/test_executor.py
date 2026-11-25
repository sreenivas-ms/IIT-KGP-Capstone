
"""Person 4 executor/analysis/chart acceptance tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from verisheet.tools.analysis import run_analysis, quarter_comparison
from verisheet.tools.chart import make_chart
from verisheet.schemas import ChartSpec, ChartType
from verisheet.sandbox import RestrictedSandbox


def test_group_aggregate_uses_requested_sum():
    """Verify that the requested SUM aggregation is applied."""
    frame = pd.DataFrame({
        "Region": ["East", "East", "West"],
        "Sales": [100, 200, 50],
    })

    result = run_analysis(
        "group_aggregate",
        {"sales": frame},
        {
            "sheet": "sales",
            "group_by": ["Region"],
            "metrics": {"Sales": "sum"},
        },
    )

    assert result.to_dict("records") == [
        {"Region": "East", "Sales": 300},
        {"Region": "West", "Sales": 50},
    ]


def test_safe_join_rejects_unexpected_many_to_many_cardinality():
    """Verify that duplicate keys are rejected for a many-to-one join."""
    left = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "sales": [100, 200, 50],
    })
    right = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "customer_name": ["A", "B", "C"],
    })

    join_spec = {
            "left_sheet": "left",
            "right_sheet": "right",
            "left_keys": ["customer_id"],
            "right_keys": ["customer_id"],
            "expected_cardinality": "many_to_one",
            "aggregate_before_join": False,
            "validate_row_count": True,
    }

    with pytest.raises(ValueError):
        run_analysis(
            "safe_join",
             {"left": left, "right": right},
             {"join_spec": join_spec},
        )


def test_quarter_comparison_uses_last_completed_quarter_policy():
    """Verify the two completed quarters before the latest observed quarter."""
    frame = pd.DataFrame({
        "date": pd.to_datetime([
            "2025-01-10",
            "2025-04-10",
            "2025-07-10",
            "2025-10-10",
        ]),
        "profit": [10, 20, 30, 40],
    })

    result = quarter_comparison(frame, "date", "profit", "sum")

    assert result["quarter"].tolist() == ["2025Q2", "2025Q3"]
    assert result["profit"].tolist() == [20.0, 30.0]
    assert result.iloc[1]["change_vs_previous"] == 10.0
    assert result.iloc[1]["percent_change"] == 50.0


def test_chart_exports_machine_readable_data_equal_to_result_table(tmp_path):
    """Verify chart JSON values originate from the result CSV."""
    result_file = tmp_path / "quarterly_profit.csv"

    result_frame = pd.DataFrame({
        "quarter": ["2025Q2", "2025Q3"],
        "profit": [20, 30],
    })
    result_frame.to_csv(result_file, index=False)

    chart_spec = ChartSpec(
        chart_type=ChartType.BAR,
        title="Quarterly Profit",
        x_column="quarter",
        y_column="profit",
    )

    output = make_chart(
        str(result_file),
        chart_spec,
        str(tmp_path / "charts"),
    )

    plotted_file = Path(output["plotted_data_path"])
    chart_file = Path(output["chart_path"])

    assert chart_file.is_file()
    assert plotted_file.is_file()

    plotted_data = json.loads(plotted_file.read_text(encoding="utf-8"))

    assert plotted_data == {
        "quarter": ["2025Q2", "2025Q3"],
        "profit": [20, 30],
    }

    assert output["plotted_data"] == plotted_data


def test_sandbox_enforces_timeout_and_no_network(tmp_path, monkeypatch):
    """Verify Docker isolation flags and timeout handling without requiring Docker."""
    import verisheet.sandbox as sandbox_module

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    script = tmp_path / "analysis.py"
    script.write_text(
        "import math\nresult = math.sqrt(16)\n",
        encoding="utf-8",
    )

    sandbox = RestrictedSandbox(timeout_seconds=2, memory_mb=256)
    sandbox.docker = "docker"

    captured = {}

    class Completed:
        returncode = 0
        stdout = "completed"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)

    result = sandbox.run_script(script, input_dir, output_dir)

    command = captured["command"]

    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert captured["kwargs"]["timeout"] == 2
    assert result["isolated"] is True
    assert result["returncode"] == 0

    def timeout_run(command, **kwargs):
        raise sandbox_module.subprocess.TimeoutExpired(
            command,
            timeout=kwargs["timeout"],
        )

    monkeypatch.setattr(sandbox_module.subprocess, "run", timeout_run)

    timeout_result = sandbox.run_script(script, input_dir, output_dir)

    assert timeout_result["timed_out"] is True
    assert timeout_result["returncode"] == -1
    assert timeout_result["isolated"] is True
