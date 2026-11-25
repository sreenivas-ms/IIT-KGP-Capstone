# P4 implementation: Executor, tools and sandbox

## Scope

P4 implements `analysis.py`, `chart.py`, `sandbox.py` and `executor.py` without changing
`schemas.py` or the shared integration signatures. The Planner supplies typed operation names
and JSON-like arguments; it never supplies executable Python.

## Supported analysis operations

- `group_aggregate`: `group_by` plus a `metrics` mapping such as `{"Sales": "sum"}`.
- `filter_rows`: typed predicates `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`,
  `contains`, `between`, `is_null`, `not_null`.
- `top_n`: `by`, `n`, optional `ascending`.
- `time_aggregate`: `date_column`, `frequency` (`month`, `quarter`, `year`), `metrics`.
- `quarter_comparison`: `date_column`, `metric_column`, optional `aggregation`.
- `safe_join`: `join_spec` with sheet names, left/right keys and expected cardinality.
- `grouped_ratio`: numerator and denominator are summed at the requested group grain before division.

The quarter comparison uses the quarter containing the maximum observed date as the incomplete
quarter and compares the two preceding quarters, matching the project default. The caller must
disclose this observed-date assumption. Percentage change is null if the previous quarter is
zero or negative.

## Sandbox setup (Windows PowerShell)

From the repository root, build the dedicated sandbox image:

```powershell
docker build -f sandbox.Dockerfile -t verisheet-sandbox:latest .
```

Docker Desktop must be running. `RestrictedSandbox.run_script()` uses `--network=none`, a
read-only root filesystem, a non-root UID, dropped Linux capabilities, no-new-privileges,
read-only input mount, writable output mount, PID/memory/CPU limits and a timeout. It fails
closed if Docker is not installed. AST checks are defense-in-depth only.

The application Executor currently compiles the Planner's typed operations into fixed Pandas
functions in-process; it does not execute generated Python. The `.py` artifact is a deterministic
operation record for audit/reproduction, not a script submitted by the model. If the integration
team requires all Pandas operations to execute inside Docker, that needs an explicit serialization
protocol for input frames and outputs and should be agreed with P5/P6 before changing the contract.

## Run P4 tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_p4_implementation.py tests/test_contracts.py
```

The full existing test suite can be run with `python -m pytest -q`. Skips owned by other
contributors remain expected until their modules are implemented.
