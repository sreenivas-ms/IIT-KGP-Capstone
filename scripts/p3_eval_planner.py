"""Live Planner evaluation against Qwen (VS-P3-02/03 acceptance evidence).

For each representative question: profile the workbook (P2) -> retrieve definitions (P2) ->
Planner.generate_plan (P3, live Qwen) -> run the plan in the real Executor (P4).
A case passes when the outcome matches the expectation (plan that executes, or clarification).

Usage (needs QWEN_* in .env; retrieval uses artifacts/index if built):
    python scripts/p3_eval_planner.py
    python scripts/p3_eval_planner.py --only margin --out results/p3_planner_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verisheet.executor import Executor  # noqa: E402
from verisheet.planner import Planner, PlanValidationError  # noqa: E402
from verisheet.profile import profile_workbook  # noqa: E402
from verisheet.retrieval import retrieve_context  # noqa: E402
from verisheet.schemas import RunStatus  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
MULTI, RETAIL = FIX / "mini_multisheet_fixture.xlsx", FIX / "online_retail_tiny.csv"

# (case id, workbook, question, expected outcome, operation the plan should use or None)
CASES = [
    ("aggregation", MULTI, "What are total sales?", "plan", "group_aggregate"),
    ("group", MULTI, "Show total sales by region", "plan", "group_aggregate"),
    ("time", MULTI, "Show monthly sales", "plan", "time_aggregate"),
    ("chart", MULTI, "Plot total profit by region as a bar chart", "plan", "group_aggregate"),
    ("margin", MULTI, "What is the profit margin by region?", "plan", "grouped_ratio"),
    ("quarter", MULTI, "Compare profit in the last two completed quarters", "plan", "quarter_comparison"),
    ("top_n", MULTI, "Which 2 orders had the highest sales?", "plan", "top_n"),
    ("join", MULTI, "Join orders with product costs on Product_ID", "plan", "safe_join"),
    ("retail_profit", RETAIL, "What was total profit?", "clarify", None),
    # Sales value = Quantity x UnitPrice needs a derived column; no executor operation computes
    # one yet (P4 gap), so the correct Planner outcome today is a clarification, not a guess.
    ("retail_sales", RETAIL, "What were total sales excluding cancellations?", "clarify", None),
    ("ambiguous", MULTI, "How did we do?", "clarify", None),
]


def run_case(planner: Planner, executor: Executor, case: tuple, workdir: Path) -> dict:
    cid, workbook, question, expected, want_op = case
    profile = profile_workbook(workbook)
    context = retrieve_context(question, profile)
    row = {"case": cid, "question": question, "expected": expected}
    try:
        plan = planner.generate_plan(question, profile, context)
    except PlanValidationError as exc:
        return {**row, "outcome": "invalid_plan", "pass": False, "detail": str(exc)[:200],
                **_usage(planner)}
    ops = [s.operation for s in plan.steps if s.tool == "run_analysis"]
    if plan.clarification_needed:
        row.update(outcome="clarify", detail=plan.clarification_question)
    else:
        results = executor.execute_plan(plan, str(workbook), f"eval-{cid}")
        ok = bool(results) and all(r.status == RunStatus.SUCCESS for r in results)
        errors = [e.message for r in results for e in r.errors]
        row.update(outcome="plan" if ok else "plan_failed_execution",
                   detail=f"ops={ops} " + ("; ".join(errors)[:160] if errors else ""))
    row["pass"] = row["outcome"] == expected and (want_op is None or want_op in ops)
    row["plan_id"], row["assumptions"] = plan.plan_id, plan.assumptions
    return {**row, **_usage(planner)}


def _usage(planner: Planner) -> dict:
    run = planner.last_run
    return {k: run.get(k) for k in ("calls", "input_tokens", "output_tokens", "latency_ms",
                                    "json_repairs", "policy_repairs")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Planner evaluation")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    planner = Planner()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        executor = Executor(artifact_dir=tmp)
        for case in CASES:
            if args.only and case[0] not in args.only:
                continue
            row = run_case(planner, executor, case, Path(tmp))
            rows.append(row)
            mark = "PASS" if row["pass"] else "FAIL"
            print(f"{mark}  {row['case']:<14} expected={row['expected']:<8} got={row['outcome']:<22} "
                  f"calls={row['calls']} repairs={row['json_repairs']}/{row['policy_repairs']}  "
                  f"{(row.get('detail') or '')[:90]}")
    passed = sum(r["pass"] for r in rows)
    print(f"\n{passed}/{len(rows)} cases passed  (prompt v{planner.prompt_version})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"saved {args.out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
