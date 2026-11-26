# VeriSheet — Agentic Spreadsheet Analysis (Starter Repository)

This repository is a **contract-first implementation scaffold** for the VeriSheet capstone. It gives six contributors clear module ownership, shared Pydantic contracts, function/class definitions, mock fixtures, test placeholders, and -style implementation stories.

> **Important:** This is intentionally not a finished application. Functions marked `####CODE HERE` are implementation work assigned to a team member. They raise `NotImplementedError` until completed. Do not mistake a successful import for a completed feature.

## Project goal

A user uploads an unfamiliar XLSX/CSV workbook and asks an analytical question. The system profiles the workbook, retrieves applicable definitions, plans an analysis with Qwen3-8B, executes approved Python/Pandas/DuckDB operations in a restricted environment, independently verifies numerical and plotted results, and returns an evidence-backed answer or a clarification/refusal.

## Model and service decisions

- **LLM:** Qwen3-8B through an OpenAI-compatible chat-completions endpoint. Planner and semantic critic use separate prompts but the same configured model endpoint.
- **Embeddings:** `BAAI/bge-small-en-v1.5` through `sentence-transformers`.
- **Vector index:** FAISS CPU (local index). Keep retrieved context grounded in source documents and cite its source identifiers.
- **Numerical verification:** deterministic Python/Pandas/DuckDB. Do not use Qwen as the authority for numeric correctness.
- **Charts:** Matplotlib or Plotly. Chart arrays must be derived from the executed result table, never generated as model-authored values.
- **UI:** Streamlit.
- **Sandbox:** separate non-privileged process/container with no network, limited resources, read-only input and a writable artifact directory. AST checks are defense-in-depth, not the security boundary.

No additional generative model is required for the baseline scope.

## Team ownership

| Person | Module | Primary files |
|---|---|---|
| P1 | Domain/data/evaluation | `scripts/`, `data/`, `docs/DATA_DICTIONARY.md`, `tests/test_data_generation.py` |
| P2 | Workbook profiling/RAG | `src/verisheet/profile.py`, `src/verisheet/retrieval.py`, `src/verisheet/tools/inspect.py` |
| P3 | Prompt engineering/Planner | `src/verisheet/planner.py`, `src/verisheet/qwen_client.py`, `prompts/planner.txt` |
| P4 | Executor/tools/sandbox | `src/verisheet/executor.py`, `src/verisheet/sandbox.py`, `src/verisheet/tools/analysis.py`, `chart.py` |
| P5 | Critic/verification/orchestration/state | `src/verisheet/critic.py`, `orchestrator.py`, `state.py`, `tools/verify.py`, `schemas.py` (shared owner; changes reviewed) |
| P6 | UI/observability/experiments/deployment | `app.py`, `src/verisheet/observability/`, `src/verisheet/experiments/`, `Dockerfile` |

All contributors must use the public contracts in `src/verisheet/schemas.py`. A change to a shared contract requires approval from every affected owner.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cp .env.example .env  # Windows: copy .env.example .env
```

Configure `QWEN_API_BASE`, `QWEN_API_KEY`, `QWEN_MODEL` (`Qwen/Qwen3-8B`) and `QWEN_CA_CERT` in `.env`.
Put the course CA certificate (for example `vm04-ca.crt`) in `config/`; `*.crt`, `.env` and key files are
git-ignored. TLS verification stays on. Never commit `.env` or credentials, and never paste a key into
prompts, traces or screenshots. If a key is exposed, ask the coordinator for a replacement.

```bash
python -m pytest -q
streamlit run app.py
```

Contract tests are executable immediately. Module-specific tests are initially marked as implementation placeholders and must be enabled as stories are completed. The Streamlit UI is a visible scaffold until Person 6 connects the orchestrator.

## Required project commands (target interfaces)

The  backlog describes implementation of these commands:

```bash
python scripts/prepare_data.py --seed 42
python scripts/generate_gold.py --config config/tasks.yaml
python experiments/run.py --system full --split dev
python experiments/run.py --system full --split test
python experiments/inject_errors.py --suite 20_cases
python experiments/score.py --results results/traces --out results/summary.csv
streamlit run app.py
```

## Non-negotiable acceptance conditions

1. XLSX with multiple sheets and CSV upload/profile, including schema and quality warnings.
2. Multi-turn analysis against the same workbook with prior **verified** metric/filter/period context retained.
3. Observable Planner → Executor → Critic loop with ACCEPT, REPAIR, REPLAN and CLARIFY transitions.
4. Executable analysis and chart artifacts, run in an isolated restricted environment.
5. Independent recomputation of numerical outputs and verification that chart data matches the verified result table.
6. Structured trace containing prompts/model versions, scripts, tool results, verdicts, token counts, latency and failure category.
7. Five system variants on the same frozen 80-question test split and at least 20 robustness/error-injection cases.

## Data and benchmark rules

The repository includes only tiny fictional development fixtures under `tests/fixtures/`; it does not bundle the public benchmark downloads, generated gold answers, or the 20 corruption workbooks. Person 1 must acquire/generate those deliverables and place them in the documented folders.

- Sample Superstore: recorded `Profit` exists, but it is not a complete accounting P&L.
- UCI Online Retail: invoice line observations and cancellations exist; **profit/COGS are not provided**. Never calculate or invent profit for this dataset.
- Synthetic FinanceOps workbook: operating profit = net sales − COGS − operating expenses. Return handling and effective-dated costs must follow `docs/DATA_DICTIONARY.md` and the dataset README.
- “Last completed quarter” defaults to the quarter before the quarter containing the maximum observed transaction date, unless an explicit `Data_Through`/reporting-close convention is available. Disclose the assumption.
- Margin is `sum(Profit) / sum(Sales)`, not the mean of row-level margins.
- Aggregate returns by `Order_ID` before joining to order lines. Validate join cardinality and row/value totals.
- Different currencies cannot be added without an exchange-rate source; clarify instead.
- Keep `data/gold/` and gold scripts outside every model prompt and retrieval index.

## Integration policy

Read `docs/INTEGRATION_CONTRACTS.md` and `docs/DEVELOPMENT_WORKFLOW.md` before coding. Develop against fixtures in `contracts/examples/` first. Do not wait for another person's module; use the agreed mock payloads. Every public function must retain its signature unless a reviewed contract change is approved.

## Documentation map

- `docs/ARCHITECTURE.md` — system flow and component boundaries.
- `docs/INTEGRATION_CONTRACTS.md` — schemas and interface rules.
- `docs/SECURITY.md` — upload, prompt-injection, PII and sandbox controls.
- `docs/DATA_DICTIONARY.md` — metric and dataset rules.
- `docs/PLANNER.md` — Qwen client, Planner prompt/policies, live evaluation and integration notes (P3).
- `docs/PROFILING_AND_RETRIEVAL.md` — profiling warnings, retrieval index build, provenance and evaluation (P2).
- `docs/CRITIC_AND_ORCHESTRATION.md` — verifier, critic routing, orchestration budgets and state (P5).

## Profiling and retrieval (P2) quick start

```bash
python scripts/build_index.py      # builds artifacts/index via the shared embedding API (EMBEDDING_* in .env)
python scripts/p2_demo.py tests/fixtures/mini_multisheet_fixture.xlsx "How is operating profit defined?"
python scripts/eval_retrieval.py   # recall@k / MRR for dense, BM25 and hybrid retrieval
```

`profile_workbook` returns a `WorkbookProfile` with coded quality warnings (for example
`HIDDEN_SHEET`, `FORMULA_NO_CACHED_VALUE`, `NUMERIC_AS_TEXT`, `DATE_COERCION_LOSS`, `POSSIBLE_PII`).
`retrieve_context` returns a `RetrievedContext` whose items carry `item_id`, `source_path`
(`file#heading`), `title`, `score` and `source_type`, plus the `corpus_version` of the index. If the
index is missing it returns no items and a `RETRIEVAL_UNAVAILABLE` warning instead of failing.

## Critic, verification and orchestration (P5) quick start

```python
from verisheet.orchestrator import AgentOrchestrator, run_agent
response = run_agent("What was profit in the last completed quarter?", "data/derived/superstore.xlsx", "session-1")
print(response.status, response.answer, response.trace_id)
```

`run_agent` returns one `AnswerResponse` on every path (VERIFIED / NOT_VERIFIED /
CLARIFICATION_REQUIRED / FAILED). Numbers are VERIFIED only after `tools/verify.py` recomputes them
from the source workbook; the Qwen critic can veto but never approve. `run_agent` wires the real
profiler, retriever, Planner and Executor; if the Qwen endpoint is not configured it returns a FAILED
response that says so. `tests/test_p5_integration.py` runs the whole loop with the real P2/P3/P4
modules and a scripted Qwen. Details, step input conventions and routing rules:
`docs/CRITIC_AND_ORCHESTRATION.md`.

## Definition of Done for every task

- Implementation merged through a reviewed pull request.
- Public functions/classes have docstrings and type hints.
- Schema and error behavior are documented.
- Unit tests cover happy path, edge cases and failure path.
- Contract test passes against the shared fixture.
- No secrets, gold answers, or raw PII are committed to prompts/logs.
- README or module documentation explains setup and usage.
- The module can be run by another contributor using the documented steps.
