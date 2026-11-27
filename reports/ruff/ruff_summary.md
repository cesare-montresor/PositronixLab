# Ruff Audit Summary (2026-11-27)

- **Tool:** `ruff 0.9.4` (configured via `pyproject.toml` `[tool.ruff]`)
- **Selected rules:** `F`, `E4`, `E7`, `E9`, `I`, `UP`, `B` (ignoring `UP038`)
- **Scope:** `positronixlab/` and `tests/`

## Overview

| Target | Total Lint Findings | Safe Auto-Fix (`--fix`) | Unsafe Auto-Fix (`--unsafe-fixes`) | Manual | Format Check (`ruff format --check`) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `positronixlab/` | 0 | 0 | 0 | 0 | 83 files would be reformatted, 30 files already formatted |
| `tests/` | 0 | 0 | 0 | 0 |  |

## `pyproject.toml` Configuration Review

Checked `[tool.ruff]` in `pyproject.toml`:

1. **Missing `ruff` in `[project.optional-dependencies].dev`**: `ruff` is currently only installed globally (`/home/cesare/.local/bin/ruff`), not declared in `dev` extras.
2. **No `exclude` list**: running `ruff check .` from repo root will also scan `libs/` (external SOTA symlinks), `draft/`, `examples/`, and `data/`. Recommend setting `extend-exclude = ["libs", "draft", "data", "papers"]`.
3. **Noise from lazy-export scaffolding & vendored code**:
   - `positronixlab/dataprovider/__init__.py` accounts for **53 of 78** `F401` hits (imports inside `if TYPE_CHECKING:` or lazy re-export scaffolding).
   - `positronixlab/baselines/lepori/_cvr_gen/subnetwork_tasks.py` (vendored 2.4k-line CVR generator) accounts for **all 76 `F405` + 1 `F403`** hits due to `from ...utils import *`.
   - Consider adding `[tool.ruff.lint.per-file-ignores]` for vendored `_cvr_gen/**` (or replacing its star import) and lazy `__init__.py` files.
4. **`line-length` not set**: defaults to `88`. Worth making explicit (e.g. `line-length = 100` or `120`) before running `ruff format` so diffs stay minimal.
5. **`UP038` in `ignore`**: valid on `ruff 0.9.4`, though deprecated/removed in newer Ruff releases (`>=0.12`).

## Rule Breakdown (`positronixlab/`)

| Rule | Count | Category | Auto-Fixable | Notes |
| :--- | ---: | :--- | :--- | :--- |

## Per-Module Breakdown (`positronixlab/`)

| Module | Total | Actionable (Logic/Bugbear/Real `F401`) | Mechanical (`UP`/`I001`/`E701`/vendored `F405`) | Top Actionable Rules |
| :--- | ---: | ---: | ---: | :--- |

## Detailed Reports

- [`ruff_positronixlab.md`](ruff_positronixlab.md) — Actionable findings in `positronixlab/` + per-file mechanical summary.
- [`ruff_tests.md`](ruff_tests.md) — Full findings breakdown for `tests/`.
