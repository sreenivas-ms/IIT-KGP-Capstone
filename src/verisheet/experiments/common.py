"""Shared helpers for the P6 experiment runner, scorer and injection suite."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "contracts" / "schemas"
DEFAULT_QUESTIONS = ROOT / "data" / "derived" / "questions.jsonl"
DEFAULT_GOLD = ROOT / "data" / "gold" / "gold_answers.jsonl"
RESULTS_ROOT = ROOT / "results" / "experiments"

# Numeric matching (documented in docs/EVALUATION.md): a gold number is matched when a candidate
# number is within an absolute 0.006 (2-decimal rounding) or a relative 1e-4.
ABS_TOL = 0.006
REL_TOL = 1e-4
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\(?\$?\d[\d,]*(?:\.\d+)?\)?%?")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")


def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads((CONTRACTS / f"{name}.schema.json").read_text()))


def load_questions(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    """Questions validated against contracts/schemas/question_task.schema.json (no gold)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Question file not found: {path}. The frozen dev/test questions are produced by "
            "Person 1 (scripts/generate_gold.py). For a pipeline check use "
            "--questions experiments/smoke/questions.jsonl")
    validator = _validator("question_task")
    rows = read_jsonl(path)
    for row in rows:
        errors = sorted(validator.iter_errors(row), key=str)
        if errors:
            raise ValueError(f"question {row.get('id')!r} violates question_task schema: {errors[0].message}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in question file")
    return [r for r in rows if split is None or r["split"] == split]


def load_gold(path: str | Path) -> dict[str, dict[str, Any]]:
    """Gold answers keyed by id, validated against gold_answer.schema.json."""
    validator = _validator("gold_answer")
    gold = {}
    for row in read_jsonl(path):
        errors = sorted(validator.iter_errors(row), key=str)
        if errors:
            raise ValueError(f"gold {row.get('id')!r} violates gold_answer schema: {errors[0].message}")
        gold[row["id"]] = row
    return gold


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata() -> dict[str, Any]:
    import numpy
    import pandas

    return {"python": platform.python_version(), "platform": platform.platform(),
            "pandas": pandas.__version__, "numpy": numpy.__version__, "git_commit": git_commit()}


# ---- numbers -------------------------------------------------------------------------------


def parse_number(text: str) -> float | None:
    """'1,250.50' -> 1250.5, '(300)' -> -300, '20%' -> 20.0; None if not a number."""
    raw = text.strip()
    negative = raw.startswith("(") and raw.endswith(")") or raw.startswith("-")
    cleaned = raw.strip("()+-$%").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def numbers_in_text(text: str | None) -> list[float]:
    values = []
    for match in _NUMBER_RE.findall(text or ""):
        value = parse_number(match)
        if value is not None:
            values.append(value)
            if match.endswith("%"):
                values.append(value / 100.0)          # "20%" also matches a gold ratio 0.2
    return values


def numbers_in_table(path: str | Path | None) -> list[float]:
    """All numeric cells of a backend result table (CSV)."""
    if not path or not Path(path).is_file():
        return []
    import pandas as pd

    frame = pd.read_csv(path)
    values = []
    for column in frame.columns:
        for cell in frame[column].tolist():
            if isinstance(cell, (int, float)) and not isinstance(cell, bool) and math.isfinite(cell):
                values.append(float(cell))
            elif isinstance(cell, str) and (number := parse_number(cell)) is not None:
                values.append(number)
    return values


def flatten_numbers(value: Any) -> list[float]:
    """Numbers inside nested dict/list structures (e.g. plotted_data, direct answers)."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(value) else []
    if isinstance(value, str):
        number = parse_number(value)
        return [number] if number is not None else []
    if isinstance(value, dict):
        return [n for v in value.values() for n in flatten_numbers(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in flatten_numbers(v)]
    return []


def gold_numbers(gold_value: Any) -> list[float]:
    return flatten_numbers(gold_value)


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * abs(b))


def all_gold_matched(gold_value: Any, candidates: list[float]) -> bool:
    """True when every gold number is matched by some candidate number (see ABS/REL_TOL)."""
    wanted = gold_numbers(gold_value)
    return bool(wanted) and all(any(close(c, g) for c in candidates) for g in wanted)
