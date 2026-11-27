"""Person 6 UI tests: upload quarantine, status styles and the Streamlit app (AppTest)."""

import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from verisheet.profile import profile_workbook
from verisheet.schemas import AnswerResponse, AnswerStatus, VerificationReport
from verisheet.ui_support import STATUS_STYLES, artifact_file, save_upload, status_style

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


# ---- upload quarantine ------------------------------------------------------------------
def test_save_upload_stores_valid_file_under_content_hash(tmp_path):
    data = (FIXTURES / "online_retail_tiny.csv").read_bytes()
    path = save_upload("../../evil name.csv", data, tmp_path / "uploads")
    assert path.parent == tmp_path / "uploads" and path.suffix == ".csv"
    assert "evil" not in path.name and len(path.stem) == 24                # user name never a path
    assert path.read_bytes() == data
    assert save_upload("again.csv", data, tmp_path / "uploads") == path      # same content, same file


@pytest.mark.parametrize("name, data, match", [
    ("book.xls", b"x", "Unsupported"),
    ("noext", b"x", "Unsupported"),
    ("empty.csv", b"", "empty"),
])
def test_save_upload_rejects_bad_files(tmp_path, name, data, match):
    with pytest.raises(ValueError, match=match):
        save_upload(name, data, tmp_path)


def test_save_upload_rejects_oversized_and_malformed_files(tmp_path, monkeypatch):
    monkeypatch.setattr("verisheet.ui_support.max_upload_bytes", lambda: 10)
    with pytest.raises(ValueError, match="limit"):
        save_upload("big.csv", b"a,b\n" * 10, tmp_path)
    monkeypatch.undo()
    macro = tmp_path / "m.xlsx"
    with zipfile.ZipFile(macro, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr("xl/vbaProject.bin", b"\0" * 10)
    with pytest.raises(ValueError, match="Macro"):
        save_upload("m.xlsx", macro.read_bytes(), tmp_path / "up")
    assert list((tmp_path / "up").iterdir()) == []                          # rejected file deleted


def test_every_status_has_a_distinct_style():
    assert set(STATUS_STYLES) == set(AnswerStatus)
    assert len({s.kind for s in STATUS_STYLES.values()}) == 4
    assert status_style("VERIFIED").kind == "success" and status_style(AnswerStatus.FAILED).kind == "error"


def test_artifact_file_only_inside_artifact_dir(tmp_path):
    inside = tmp_path / "artifacts" / "run" / "t.csv"
    inside.parent.mkdir(parents=True)
    inside.write_text("a\n1\n")
    assert artifact_file(str(inside), tmp_path / "artifacts") == inside.resolve()
    assert artifact_file(str(ROOT / "README.md"), tmp_path / "artifacts") is None
    assert artifact_file(None, tmp_path) is None


# ---- Streamlit app (headless) -----------------------------------------------------------
@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """App with a loaded workbook and a fake backend that returns a VERIFIED answer."""
    monkeypatch.chdir(tmp_path)
    table = tmp_path / "artifacts" / "run-1" / "by-region.csv"
    table.parent.mkdir(parents=True)
    table.write_text("Region,Sales\nEast,50\nWest,90\n")
    calls = []

    def fake_run_agent(question, workbook, session_id, orchestrator=None):
        calls.append((question, workbook, session_id))
        report = VerificationReport(verification_id="verify-ui", verified=True, numeric_checks_passed=True,
                                    chart_checks_passed=True, semantic_checks_passed=True)
        return AnswerResponse(status=AnswerStatus.VERIFIED, answer="Sales by Region: East: 50; West: 90.",
                              assumptions=["Sales column of Orders"], result_table_path=str(table),
                              verification=report, trace_id="trace-ui-test")

    monkeypatch.setattr("verisheet.orchestrator.build_default_orchestrator", lambda: object())
    monkeypatch.setattr("verisheet.orchestrator.run_agent", fake_run_agent)
    workbook = FIXTURES / "mini_multisheet_fixture.xlsx"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["workbook_path"] = str(workbook)
    at.session_state["workbook_name"] = workbook.name
    at.session_state["profile"] = profile_workbook(workbook).model_dump(mode="json")
    at.session_state["session_id"] = "ui-test-session"
    return at, calls


def test_app_without_workbook_asks_for_upload():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("Upload a workbook" in i.value for i in at.info)
    assert not at.chat_input                                                   # no chat before upload


def test_app_shows_profile_and_renders_verified_answer_from_backend(app_env):
    at, calls = app_env
    at.run()
    assert not at.exception
    assert any("Orders" in str(df.value) for df in at.dataframe)               # profile shown
    at.chat_input[0].set_value("Total sales by region").run()
    assert not at.exception
    assert calls == [("Total sales by region", str(FIXTURES / "mini_multisheet_fixture.xlsx"), "ui-test-session")]
    assert any("VERIFIED" in s.value for s in at.success)
    assert any("trace-ui-test" in c.value for c in at.caption)                 # trace ID visible
    assert any(list(df.value.columns) == ["Region", "Sales"] for df in at.dataframe)   # backend table


def test_follow_up_keeps_session_and_workbook(app_env):
    at, calls = app_env
    at.run()
    at.chat_input[0].set_value("Total sales by region").run()
    at.chat_input[0].set_value("Now only East").run()
    assert not at.exception
    assert len(calls) == 2 and calls[0][1:] == calls[1][1:]                    # same workbook + session
    assert len(at.session_state["history"]) == 2


def test_backend_crash_shows_failed_status_not_a_stack_trace(app_env, monkeypatch):
    at, _ = app_env

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("verisheet.orchestrator.run_agent", boom)
    at.run()
    at.chat_input[0].set_value("anything").run()
    assert not at.exception
    assert any("FAILED" in e.value for e in at.error)
    assert not any("secret internal detail" in m.value for m in at.markdown)


def test_trace_page_renders_with_no_traces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["page"] = "Traces"
    at.run()
    assert not at.exception and any("No traces yet" in i.value for i in at.info)
