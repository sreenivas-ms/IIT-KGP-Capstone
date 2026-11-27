# 10-minute demo runbook

**Goal:** show a live, verified, multi-turn analysis; a refusal; one deliberate fault with a
Critic transition; and the trace dashboard showing real execution evidence.

## Before the demo (T−15 min)

```bash
cd IIT-KGP-Capstone
.venv/bin/python scripts/p3_api_check.py          # Qwen reachable (status 200, JSON reply)
.venv/bin/python scripts/build_index.py           # retrieval index via the embedding API
.venv/bin/python -m pytest -q                     # all green
.venv/bin/streamlit run app.py                    # opens http://localhost:8501
```

- Keep `tests/fixtures/mini_multisheet_fixture.xlsx` and `tests/fixtures/online_retail_tiny.csv`
  open in Finder for uploading. Use the Superstore and Online Retail files instead once Person 1
  delivers them.
- Run one throwaway question first, so the model and caches are warm.
- Fallback if the Qwen endpoint is down (HTTP 503): show the recorded traces in the Traces page
  and the results in `results/experiments/<run>/`. Say clearly that they are recorded runs.

## Script

| Time | Do | Say / point at |
|---|---|---|
| 0:00–1:00 | Title, then the architecture diagram (report §2) | Profile → retrieve → plan (Qwen) → execute (typed pandas) → **independent verification** → critic → answer. The model plans and never computes. |
| 1:00–2:30 | Upload `mini_multisheet_fixture.xlsx` | Profile: 4 sheets, roles (identifier / measure / date), the `NEGATIVE_VALUES` warning, and `EMPTY_SHEET` on README. Uploads are hashed and checked (size, macros, zip bombs). |
| 2:30–4:00 | Ask: *Compare profit in the last two completed quarters* | **VERIFIED** badge, table and chart from backend files, and the assumption about the last completed quarter. |
| 4:00–5:00 | Follow-up: *Now break it down by region* | Same session; the verified quarter is kept (East 4, West 6, not all-time totals). |
| 5:00–6:00 | Upload `online_retail_tiny.csv` and ask *What was total profit?* | **CLARIFICATION NEEDED**: the dataset has no cost data, so the system never substitutes sales for profit. |
| 6:00–8:00 | Deliberate fault: in a terminal, run `.venv/bin/python experiments/inject_errors.py --suite demo --only c01 c10` | The aggregation is swapped to mean, then the chart values are tampered with. The verifier recomputes independently → Critic **REPAIR** → re-execution → VERIFIED. The output shows `fired=1 … PASS`. |
| 8:00–9:30 | Traces page, then open the c01 trace | Timeline: planner tokens, latency, prompt v1.5, `executor:execute_step`, `verifier:verify FAIL`, `critic_decision REPAIR`, transition, second execution, `verify PASS`. Script hash and chart artifact. |
| 9:30–10:00 | `results/experiments/<run>/ablation.md` | E1–E5 on identical questions: what the planner, the verification and the critic each add. Note that these are smoke-set numbers until the frozen test set exists. |

## Checklist for the acceptance criteria

- [ ] Original Superstore and Online Retail scenarios, once Person 1's data exists. Until then,
      use the fixtures that stand in for them.
- [ ] Follow-up context retained (4:00).
- [ ] One deliberate fault and Critic transition (6:00).
- [ ] Trace dashboard shows actual execution, scripts, verification, tokens and latency (8:00).
