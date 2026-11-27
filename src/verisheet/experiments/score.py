"""Compute evaluation metrics from structured run records and gold records (VS-P6-04).

Metric definitions (also in docs/EVALUATION.md). A = answerable questions (must_refuse=false),
R = refusal questions (must_refuse=true), V = records with status VERIFIED.

numeric_accuracy        share of A whose gold numbers are all matched by numbers the system
                        computed (executed result tables, chart, answer; |diff| <= max(0.006, 1e-4*|gold|))
delivered_numeric_accuracy  same, using only what the user is shown (answer text, response table,
                        chart). E2/E3 compute but withhold unverified numbers, so this differs.
task_success            A: numeric match and status VERIFIED/NOT_VERIFIED; R: CLARIFICATION_REQUIRED
execution_success       share of A that produced an answer (status VERIFIED/NOT_VERIFIED, no crash)
plot_data_accuracy      share of charted A records whose plotted values contain every gold number
chart_appropriateness   share of charted records whose chart type is in accepted_charts
refusal_accuracy        share of R answered with CLARIFICATION_REQUIRED
false_verification_rate share of V that are not task successes (wrong but labelled VERIFIED)
verified_coverage       share of A that ended VERIFIED (and verified_correct_coverage: VERIFIED and correct)
repair_success          share of records with >=1 repair/replan that ended as task successes
tool_efficiency         mean agent actions and LLM calls per question
latency / tokens        median and p95 per question; total tokens
error_detection_recall  (injection suite) share of triggered faults that did not end VERIFIED-and-wrong
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from verisheet.experiments.common import (
    DEFAULT_GOLD,
    all_gold_matched,
    load_gold,
    read_jsonl,
    sha256_file,
    utc_now,
)

ANSWER_STATUSES = {"VERIFIED", "NOT_VERIFIED"}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def judge(record: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Per-question verdicts used by every aggregate metric."""
    refuse = bool(gold["must_refuse"])
    shown = [*record.get("table_numbers", []), *record.get("plotted_numbers", []), *record.get("answer_numbers", [])]
    candidates = [*shown, *record.get("executed_numbers", [])]
    status = record["status"]
    numeric = (not refuse) and gold.get("gold_value") is not None and all_gold_matched(gold["gold_value"], candidates)
    delivered = (not refuse) and gold.get("gold_value") is not None and all_gold_matched(gold["gold_value"], shown)
    success = status == "CLARIFICATION_REQUIRED" if refuse else (numeric and status in ANSWER_STATUSES)
    charted = bool(record.get("chart_type"))
    plot_ok = (charted and not refuse and gold.get("gold_value") is not None
               and all_gold_matched(gold["gold_value"], record.get("plotted_numbers", [])))
    accepted = gold.get("accepted_charts") or []
    return {
        "question_id": record["question_id"], "must_refuse": refuse, "status": status,
        "numeric_correct": numeric, "delivered_correct": delivered, "task_success": success,
        "executed": (not refuse) and status in ANSWER_STATUSES and not record.get("error"),
        "charted": charted, "plot_correct": plot_ok,
        "chart_appropriate": (record.get("chart_type") in accepted) if charted and accepted else None,
        "verified": status == "VERIFIED", "repaired": (record.get("repairs", 0) + record.get("replans", 0)) > 0,
    }


def score_records(records: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics for one variant's records."""
    missing = [r["question_id"] for r in records if r["question_id"] not in gold]
    if missing:
        raise ValueError(f"no gold for question ids {missing[:5]}")
    verdicts = [judge(r, gold[r["question_id"]]) for r in records]
    answerable = [v for v in verdicts if not v["must_refuse"]]
    refusals = [v for v in verdicts if v["must_refuse"]]
    verified = [v for v in verdicts if v["verified"]]
    charted = [v for v in answerable if v["charted"]]
    chart_judged = [v for v in verdicts if v["chart_appropriate"] is not None]
    repaired = [v for v in verdicts if v["repaired"]]
    latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    tokens = [float(r.get("input_tokens", 0) + r.get("output_tokens", 0)) for r in records]
    actions = [float(r["actions"]) for r in records if r.get("actions") is not None]
    return {
        "n_questions": len(verdicts), "n_answerable": len(answerable), "n_refusal": len(refusals),
        "numeric_accuracy": _rate(sum(v["numeric_correct"] for v in answerable), len(answerable)),
        "delivered_numeric_accuracy": _rate(sum(v["delivered_correct"] for v in answerable), len(answerable)),
        "task_success": _rate(sum(v["task_success"] for v in verdicts), len(verdicts)),
        "execution_success": _rate(sum(v["executed"] for v in answerable), len(answerable)),
        "plot_data_accuracy": _rate(sum(v["plot_correct"] for v in charted), len(charted)),
        "chart_appropriateness": _rate(sum(v["chart_appropriate"] for v in chart_judged), len(chart_judged)),
        "refusal_accuracy": _rate(sum(v["status"] == "CLARIFICATION_REQUIRED" for v in refusals), len(refusals)),
        "false_verification_rate": _rate(sum(not v["task_success"] for v in verified), len(verified)),
        "verified_coverage": _rate(sum(v["verified"] for v in answerable), len(answerable)),
        "verified_correct_coverage": _rate(sum(v["verified"] and v["task_success"] for v in answerable),
                                           len(answerable)),
        "n_verified": len(verified),
        "repair_success": _rate(sum(v["task_success"] for v in repaired), len(repaired)),
        "n_repaired": len(repaired),
        "mean_actions": round(statistics.mean(actions), 2) if actions else None,
        "mean_llm_calls": round(statistics.mean(r.get("llm_calls", 0) for r in records), 2) if records else None,
        "latency_ms_median": _percentile(latencies, 0.5), "latency_ms_p95": _percentile(latencies, 0.95),
        "tokens_median": _percentile(tokens, 0.5), "tokens_p95": _percentile(tokens, 0.95),
        "tokens_total": int(sum(tokens)),
        "crashes": sum(bool(r.get("error")) for r in records),
        "per_question": verdicts,
    }


def _run_dirs(results: str | Path) -> list[Path]:
    root = Path(results)
    if (root / "results.jsonl").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("results.jsonl"))


def score_results(results_dir: str, gold_file: str, output_path: str) -> dict:
    """Calculate numeric accuracy, task success, execution success, plot accuracy,
    chart appropriateness, error-detection recall, false-verification rate, verified coverage,
    repair success, tool efficiency and median/p95 latency/token usage.

    ``results_dir`` may be one run directory or a folder of runs (one per variant). If an
    ``injection_results.jsonl`` is found, error-detection recall is included.
    """
    gold_path = Path(gold_file)
    if gold_path.is_dir():
        gold_path = gold_path / "gold_answers.jsonl"
    gold = load_gold(gold_path)
    summary: dict[str, Any] = {"scored_at": utc_now(), "gold_file": str(gold_path),
                               "gold_sha256": sha256_file(gold_path), "variants": {}}
    for run_dir in _run_dirs(results_dir):
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        records = read_jsonl(run_dir / "results.jsonl")
        metrics = score_records(records, gold)
        summary["variants"][meta["variant"]] = {
            "variant_label": meta.get("variant_label"), "run_id": meta["run_id"], "split": meta["split"],
            "test_set_sha256": meta["test_set_sha256"], "model_name": meta["model_name"],
            "prompt_versions": meta["prompt_versions"], "git_commit": meta.get("git_commit"), **metrics}
    hashes = {v["test_set_sha256"] for v in summary["variants"].values()}
    summary["same_test_set"] = len(hashes) <= 1
    summary["test_set_sha256"] = next(iter(hashes)) if len(hashes) == 1 else sorted(hashes)
    injection = next(Path(results_dir).rglob("injection_results.jsonl"), None) if Path(results_dir).is_dir() else None
    if injection:
        from verisheet.experiments.inject_errors import summarize_injection

        summary["injection"] = summarize_injection(read_jsonl(injection))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint for scoring experiment traces."""
    parser = argparse.ArgumentParser(description="Score experiment runs against gold answers")
    parser.add_argument("--results", required=True, help="a run dir or a folder of run dirs")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="gold_answer JSONL (or its folder)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = score_results(args.results, args.gold, args.out)
    keys = ("numeric_accuracy", "task_success", "verified_coverage", "false_verification_rate",
            "refusal_accuracy", "latency_ms_median", "tokens_median")
    print(f"{'variant':<22}" + "".join(f"{k[:16]:>18}" for k in keys))
    for name, metrics in summary["variants"].items():
        print(f"{metrics['variant_label'] + ' ' + name:<22}" + "".join(f"{str(metrics[k]):>18}" for k in keys))
    print(f"saved {args.out}  (same test set: {summary['same_test_set']})")


if __name__ == "__main__":
    main()
