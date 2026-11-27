"""Testable helpers for the Streamlit UI (Person 6). No analytics are computed here.

- ``save_upload``: quarantines an uploaded workbook under a content-hash name (the user's file
  name never becomes a path) and validates it with the P2 upload checks.
- ``STATUS_STYLES`` / ``status_style``: how each AnswerStatus is shown.
- ``artifact_file``: resolves backend artifact paths only inside the artifact directory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from verisheet.observability.trace_dashboard import safe_artifact
from verisheet.profile import max_upload_bytes, validate_upload
from verisheet.schemas import AnswerStatus

ALLOWED_EXTENSIONS = {".xlsx", ".csv"}


@dataclass(frozen=True)
class StatusStyle:
    label: str
    kind: str        # streamlit alert: success | warning | info | error
    icon: str
    meaning: str


STATUS_STYLES: dict[AnswerStatus, StatusStyle] = {
    AnswerStatus.VERIFIED: StatusStyle(
        "VERIFIED", "success", "✅",
        "Every number was independently recomputed from the workbook and the chart was checked."),
    AnswerStatus.NOT_VERIFIED: StatusStyle(
        "NOT VERIFIED", "warning", "⚠️",
        "An answer was produced but could not be independently verified. Do not rely on it."),
    AnswerStatus.CLARIFICATION_REQUIRED: StatusStyle(
        "CLARIFICATION NEEDED", "info", "❓",
        "More information is needed before answering without guessing."),
    AnswerStatus.FAILED: StatusStyle(
        "FAILED", "error", "⛔", "No answer could be produced."),
}


def status_style(status: AnswerStatus | str) -> StatusStyle:
    return STATUS_STYLES[AnswerStatus(status)]


def save_upload(file_name: str, data: bytes, upload_root: str | Path) -> Path:
    """Store an uploaded workbook safely and return its path.

    Raises:
        ValueError: unsupported extension, empty/oversized file, or a file that fails the
            P2 container checks (malformed XLSX, macros, zip bomb). Rejected files are deleted.
    """
    suffix = Path(file_name or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix or '(none)'}'. Upload an .xlsx or .csv file.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    limit = max_upload_bytes()
    if len(data) > limit:
        raise ValueError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {limit / 1e6:.0f} MB.")
    digest = hashlib.sha256(data).hexdigest()
    folder = Path(upload_root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest[:24]}{suffix}"
    if not path.exists():
        path.write_bytes(data)
        path.chmod(0o600)
    try:
        validate_upload(path)
    except ValueError:
        path.unlink(missing_ok=True)
        raise
    return path


def artifact_file(path: str | None, artifact_root: str | Path) -> Path | None:
    """Backend artifact path, only if it exists inside the artifact directory."""
    return safe_artifact(path, artifact_root) if path else None
