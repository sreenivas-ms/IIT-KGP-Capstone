"""Build the P6 smoke question set and its gold file from the repository fixtures.

Usage:  python experiments/build_smoke_set.py   (writes experiments/smoke/{questions,gold}.jsonl)

This is a small development set to exercise the five-variant runner, scorer and dashboard
end to end. It is NOT the frozen 80-question held-out test set from Person 1; results on it
must be reported as smoke results. Questions and gold go to separate files: the runner reads
only questions, the scorer reads gold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verisheet.experiments.gold import MINI, RETAIL, ROOT, compute_gold

SMOKE_DIR = ROOT / "experiments" / "smoke"

# (id, workbook, question, family, gold function or None, must_refuse, accepted charts)
SMOKE_QUESTIONS = [
    ("s01", MINI, "What were total sales?", "aggregation_and_ranking", "total_sales", False, ["table", "bar"]),
    ("s02", MINI, "Show total sales by region as a bar chart.", "aggregation_and_ranking", "sales_by_region",
     False, ["bar", "horizontal_bar"]),
    ("s03", MINI, "What was total profit by region?", "grouping_and_multisheet_joins", "profit_by_region",
     False, ["bar", "horizontal_bar", "table"]),
    ("s04", MINI, "What is the profit margin by region?", "multistep_explanations", "margin_by_region",
     False, ["bar", "horizontal_bar", "table"]),
    ("s05", MINI, "Show monthly sales as a line chart.", "time_periods_and_growth", "monthly_sales",
     False, ["line"]),
    ("s06", MINI, "Compare profit in the last two completed quarters.", "time_periods_and_growth",
     "last_two_quarters_profit", False, ["bar", "table"]),
    ("s07", MINI, "Which order had the highest sales?", "aggregation_and_ranking", "top_order_by_sales",
     False, ["bar", "horizontal_bar", "table"]),
    ("s08", MINI, "How many orders were placed in each region?", "grouping_and_multisheet_joins",
     "order_count_by_region", False, ["bar", "horizontal_bar", "table"]),
    ("s09", RETAIL, "What was total profit?", "ambiguity_corruption_refusal", None, True, []),
    ("s10", MINI, "How did we do?", "ambiguity_corruption_refusal", None, True, []),
    ("s11", RETAIL, "How many invoice lines are there per country?", "grouping_and_multisheet_joins",
     "retail_invoice_lines_by_country", False, ["bar", "horizontal_bar", "table"]),
    ("s12", RETAIL, "What was the total quantity sold, excluding cancellations (negative quantities)?",
     "aggregation_and_ranking", "retail_quantity_excluding_cancellations", False, ["table", "bar"]),
]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def build_smoke_set(out_dir: str | Path = SMOKE_DIR, split: str = "dev") -> dict[str, str]:
    """Write questions.jsonl (contract: question_task) and gold.jsonl (contract: gold_answer)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    questions, gold = [], []
    for qid, workbook, question, family, gold_fn, refuse, charts in SMOKE_QUESTIONS:
        digest = sha256_file(workbook)
        questions.append({"id": qid, "workbook": workbook, "workbook_sha256": digest, "question": question,
                          "family": family, "split": split, "must_refuse": refuse, "accepted_charts": charts,
                          "notes": "P6 smoke set over repository fixtures (not the frozen test set)."})
        gold.append({"id": qid, "workbook_sha256": digest, "gold_value": compute_gold(gold_fn, workbook),
                     "accepted_charts": charts, "must_refuse": refuse,
                     "gold_script": f"verisheet.experiments.gold.{gold_fn}" if gold_fn else None,
                     "notes": "computed with pandas from the workbook; never hand-entered"})
    q_path, g_path = out / "questions.jsonl", out / "gold.jsonl"
    q_path.write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    g_path.write_text("".join(json.dumps(g) + "\n" for g in gold), encoding="utf-8")
    return {"questions": str(q_path), "gold": str(g_path)}
