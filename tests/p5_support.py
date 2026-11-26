"""Shared fakes and workbook builders for Person 5 critic/verifier/orchestrator tests.

Everything here is synthetic. The fake Executor writes real CSV artifacts so the verifier is
exercised end-to-end, and it can inject the faults described in the capstone spec
(wrong aggregation, wrong quarter, altered plot value, dropped category, wrong chart type).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from verisheet.schemas import (
    AnalysisPlan, AnalysisStep, ChartSpec, ExecutionResult, LLMResponse, RunStatus, ToolError,
)
from verisheet.tools import verify as V

# Hand-computable data. Max date 2024-10-01 -> last completed quarter 2024Q3.
# Q2 profit = 30 - 40 + 50 = 40; Q3 profit = 60 + 70 = 130.
ORDERS = pd.DataFrame({
    "Order_ID": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"],
    "Order_Date": pd.to_datetime(["2024-01-15", "2024-02-10", "2024-04-05", "2024-05-20",
                                  "2024-06-30", "2024-07-10", "2024-08-15", "2024-10-01"]),
    "Region": ["East", "West", "East", "West", "East", "West", "East", "West"],
    "Category": ["A", "B", "A", "B", "C", "A", "B", "C"],
    "Sales": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
    "Profit": [10.0, 20.0, 30.0, -40.0, 50.0, 60.0, 70.0, 80.0],
})
RETURNS = pd.DataFrame({
    "Return_ID": ["R1", "R2", "R3"],
    "Order_ID": ["O2", "O2", "O4"],  # O2 returned twice -> non-unique join key
    "Refund_Amount": [5.0, 7.0, 10.0],
})


def write_superstore(path: Path, *, data_through: str | None = None) -> Path:
    """Write a two-sheet workbook (Orders, Returns) and optionally a Definitions sheet."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        ORDERS.to_excel(writer, sheet_name="Orders", index=False)
        RETURNS.to_excel(writer, sheet_name="Returns", index=False)
        if data_through:
            pd.DataFrame({"Metric": ["profit"], "Data_Through": [pd.Timestamp(data_through)]}).to_excel(
                writer, sheet_name="Definitions", index=False)
    return path


def write_retail_csv(path: Path) -> Path:
    """Online-Retail-like CSV: sales fields only, no profit/cost columns."""
    pd.DataFrame({
        "InvoiceNo": ["536365", "536366", "C536379", "536370"],
        "StockCode": ["85123A", "71053", "D", "22728"],
        "Quantity": [6, 6, -1, 24],
        "InvoiceDate": ["2010-12-01 08:26", "2010-12-01 08:28", "2010-12-01 09:41", "2011-01-05 08:45"],
        "UnitPrice": [2.55, 3.39, 27.5, 3.75],
        "CustomerID": [17850, 17850, 14527, 12583],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "France"],
    }).to_csv(path, index=False)
    return path


def quarter_plan(plan_id: str = "plan-q", *, chart_type: str = "bar", date_range: dict | None = None,
                 policy: str = "previous_to_quarter_containing_max_observed_date",
                 aggregation: str = "sum", quarter: str | None = None) -> AnalysisPlan:
    """S01-style plan: completed-quarter profit comparison plus a chart."""
    inputs: dict[str, Any] = {"sheet": "Orders", "date_column": "Order_Date", "metric_column": "Profit",
                              "aggregation": aggregation}
    if quarter:
        inputs["quarter"] = quarter
    return AnalysisPlan(
        plan_id=plan_id, goal="compare_latest_completed_quarters_profit",
        metric={"name": "profit", "source_columns": ["Profit"], "aggregation": aggregation,
                "formula": "sum(Profit)", "unit": "currency"},
        source_sheets=["Orders"], date_policy=policy, date_range=date_range,
        steps=[
            AnalysisStep(step_id="agg", tool="run_analysis", operation="quarter_comparison", inputs=inputs),
            AnalysisStep(step_id="chart", tool="make_chart", operation=chart_type,
                         inputs={"source_step_id": "agg"}, depends_on=["agg"]),
        ],
        chart_spec=ChartSpec(chart_type=chart_type, title="Quarterly profit", x_column="quarter",
                             y_column="profit", unit="currency", sort="chronological", baseline=0.0),
        assumptions=["Last completed quarter = quarter before the one containing the max order date."],
    )


def ranking_plan(plan_id: str = "plan-rank", chart_type: str = "horizontal_bar", n: int = 2) -> AnalysisPlan:
    """Top-N categories by total sales."""
    return AnalysisPlan(
        plan_id=plan_id, goal="top_categories_by_sales",
        metric={"name": "sales", "source_columns": ["Sales"], "aggregation": "sum", "unit": "currency"},
        source_sheets=["Orders"], group_by=["Category"],
        steps=[
            AnalysisStep(step_id="rank", tool="run_analysis", operation="top_n",
                         inputs={"sheet": "Orders", "group_by": ["Category"], "metric_column": "Sales",
                                 "aggregation": "sum", "n": n}),
            AnalysisStep(step_id="chart", tool="make_chart", operation=chart_type,
                         inputs={"source_step_id": "rank"}, depends_on=["rank"]),
        ],
        chart_spec=ChartSpec(chart_type=chart_type, title="Top categories", x_column="Category",
                             y_column="sales", sort="descending", baseline=0.0),
    )


class FakeExecutor:
    """Executor double that writes real artifacts and injects configurable faults.

    faults: list of per-call fault sets; call N of execute_plan/execute_step uses faults[N]
    (or no fault once the list is exhausted). Supported faults: "mean", "shift_quarter",
    "alter_plot", "drop_category", "fail_chart", "raise".
    """

    def __init__(self, artifact_dir: Path, faults: list[set[str]] | None = None, always: set[str] | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.faults = list(faults or [])
        self.always = set(always or set())
        self.calls: list[tuple[str, list[str]]] = []
        self.tables: dict[str, pd.DataFrame] = {}

    def _fault(self) -> set[str]:
        current = self.faults.pop(0) if self.faults else set()
        return current | self.always

    def execute_plan(self, plan: AnalysisPlan, workbook_path: str, run_id: str) -> list[ExecutionResult]:
        fault = self._fault()
        self.calls.append(("plan", [s.step_id for s in plan.steps]))
        if "raise" in fault:
            raise RuntimeError("boom")
        prior: dict[str, ExecutionResult] = {}
        for step in plan.steps:
            prior[step.step_id] = self._run(step, plan, workbook_path, run_id, fault)
        return list(prior.values())

    def execute_step(self, step, workbook_path, run_id, prior_results) -> ExecutionResult:
        fault = self._fault()
        self.calls.append(("step", [step.step_id]))
        return self._run(step, self._plan, workbook_path, run_id, fault)

    def _run(self, step, plan, workbook_path, run_id, fault) -> ExecutionResult:
        self._plan = plan
        out_dir = self.artifact_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if step.tool == "run_analysis":
            frames = V.load_source_frames(workbook_path)
            canonical = V._canonical(step.operation, step, plan, frames)
            table = canonical.table.copy()
            value_col = canonical.value_columns[0]
            if "mean" in fault and canonical.data is not None:
                src = canonical.metric_columns[value_col]
                if canonical.group_columns:
                    alt = canonical.data.groupby(canonical.group_columns)[src].mean()
                    table[value_col] = [alt[k] for k in table[canonical.group_columns[0]]]
            if "shift_quarter" in fault and "quarter" in table.columns:
                table["quarter"] = [str(pd.Period(q, freq="Q") - 1) for q in table["quarter"]]
            path = out_dir / f"{step.step_id}.csv"
            table.to_csv(path, index=False)
            self.tables[step.step_id] = table
            return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.SUCCESS,
                                   result_table_path=str(path), result_columns=list(table.columns),
                                   row_count_in=canonical.row_count_in, row_count_out=len(table),
                                   script_path=str(out_dir / f"{step.step_id}.py"), script_hash="sha256:test",
                                   tool_calls=["run_analysis"])
        if "fail_chart" in fault:
            return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.FAILED,
                                   errors=[ToolError(code="CHART_RENDER", category="runtime",
                                                     message="renderer crashed", retryable=True)])
        source_id = step.inputs.get("source_step_id") or step.depends_on[0]
        table = self.tables[source_id]
        spec = plan.chart_spec
        x = [str(v) for v in table[spec.x_column]] if spec.x_column in table.columns else [str(v) for v in table.iloc[:, 0]]
        y_col = next((c for c in table.columns if V._norm(c) == V._norm(spec.y_column)), table.columns[-1])
        y = [float(v) for v in table[y_col]]
        if "alter_plot" in fault:
            y[-1] = y[-1] + 1000.0
        if "drop_category" in fault:
            x, y = x[:-1], y[:-1]
        return ExecutionResult(run_id=run_id, step_id=step.step_id, status=RunStatus.SUCCESS,
                               result_table_path=str(out_dir / f"{source_id}.csv"),
                               chart_path=str(out_dir / f"{step.step_id}.png"), chart_spec=spec,
                               plotted_data={"x": x, "y": y}, tool_calls=["make_chart"])


class ScriptedPlanner:
    """Planner double returning queued plans and recording feedback."""

    def __init__(self, *plans: AnalysisPlan, error: Exception | None = None):
        self.plans = list(plans)
        self.error = error
        self.generate_calls: list[dict] = []
        self.revise_calls: list[dict] = []
        self.prompt_version = "test"

    def generate_plan(self, question, workbook_profile, retrieved_context, previous_state=None, feedback=None):
        self.generate_calls.append({"question": question, "previous_state": previous_state, "feedback": feedback})
        if self.error:
            raise self.error
        return self.plans.pop(0)

    def revise_plan(self, prior_plan, critic_feedback, state):
        self.revise_calls.append({"prior": prior_plan.plan_id, "feedback": critic_feedback})
        return self.plans.pop(0)


class FakeQwen:
    """QwenClient double returning scripted contents."""

    def __init__(self, *contents: str, error: Exception | None = None):
        self.contents = list(contents)
        self.error = error
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system_prompt, user_prompt, temperature=0.0, max_tokens=None):
        self.prompts.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return LLMResponse(content=self.contents.pop(0) if self.contents else "{}", model="Qwen/Qwen3-8B",
                           input_tokens=100, output_tokens=20, latency_ms=5.0)


class ListTraceLogger:
    """In-memory trace sink."""

    def __init__(self):
        self.events = []

    def log_event(self, event):
        self.events.append(event)
