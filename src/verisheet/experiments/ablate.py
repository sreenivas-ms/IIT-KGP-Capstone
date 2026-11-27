"""Ablation helper for consistent E1–E5 comparison.

Usage:  python experiments/ablate.py --runs results/experiments/smoke-<ts> --gold experiments/smoke/gold.jsonl \\
            --out results/experiments/smoke-<ts>/ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verisheet.experiments.common import DEFAULT_GOLD, load_gold, read_jsonl
from verisheet.experiments.score import score_records

ORDER = ["direct", "planner_executor", "llm_critic", "deterministic_critic", "full"]
HEADLINE = ["task_success", "numeric_accuracy", "delivered_numeric_accuracy", "verified_coverage", "verified_correct_coverage",
            "false_verification_rate", "refusal_accuracy", "plot_data_accuracy", "chart_appropriateness",
            "mean_llm_calls", "latency_ms_median", "tokens_median"]


def _load(path: str | Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    run_dir = path if path.is_dir() else path.parent
    return json.loads((run_dir / "run_metadata.json").read_text()), read_jsonl(run_dir / "results.jsonl")


def compare_variants(result_files: list[str], output_path: str, gold_file: str | Path = DEFAULT_GOLD) -> dict:
    """Compare five system variants on identical question IDs and summarize metric deltas."""
    gold = load_gold(gold_file)
    runs = {}
    for item in result_files:
        meta, records = _load(item)
        runs[meta["variant"]] = (meta, records)
    id_sets = {v: tuple(sorted(r["question_id"] for r in recs)) for v, (_, recs) in runs.items()}
    hashes = {meta["test_set_sha256"] for meta, _ in runs.values()}
    if len(set(id_sets.values())) > 1 or len(hashes) > 1:
        raise ValueError("variants were not run on identical question ids / test set; comparison refused")

    metrics = {v: score_records(recs, gold) for v, (_, recs) in runs.items()}
    baseline = metrics.get("full")
    table = []
    for variant in [v for v in ORDER if v in metrics]:
        row: dict[str, Any] = {"variant": variant, "label": runs[variant][0].get("variant_label")}
        for key in HEADLINE:
            value = metrics[variant].get(key)
            row[key] = value
            if baseline and variant != "full" and isinstance(value, (int, float)) and \
                    isinstance(baseline.get(key), (int, float)):
                row[f"{key}_vs_full"] = round(value - baseline[key], 4)
        table.append(row)
    per_question = {}
    for variant, m in metrics.items():
        for verdict in m["per_question"]:
            per_question.setdefault(verdict["question_id"], {})[variant] = {
                "status": verdict["status"], "success": verdict["task_success"]}
    result = {"question_ids": list(next(iter(id_sets.values()))), "test_set_sha256": next(iter(hashes)),
              "model_names": sorted({m["model_name"] for m, _ in runs.values()}),
              "prompt_versions": {v: m["prompt_versions"] for v, (m, _) in runs.items()},
              "table": table, "per_question": per_question}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(table), encoding="utf-8")
    return result


def to_markdown(table: list[dict[str, Any]]) -> str:
    keys = ["task_success", "numeric_accuracy", "delivered_numeric_accuracy", "verified_correct_coverage", "false_verification_rate",
            "refusal_accuracy", "mean_llm_calls", "latency_ms_median", "tokens_median"]
    lines = ["| Variant | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for row in table:
        lines.append(f"| {row['label']} {row['variant']} | " + " | ".join(
            "—" if row.get(k) is None else str(row[k]) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare E1–E5 runs on identical questions")
    parser.add_argument("--runs", required=True, help="folder containing one run dir per variant")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    files = sorted(str(p.parent) for p in Path(args.runs).rglob("results.jsonl"))
    result = compare_variants(files, args.out, args.gold)
    print(to_markdown(result["table"]))
    print(f"saved {args.out} and {Path(args.out).with_suffix('.md')}")
