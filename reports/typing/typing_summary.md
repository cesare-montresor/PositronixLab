# Type-Checking Audit Summary (2026-11-27)

- **Mypy:** `mypy 2.4.1 (compiled: yes)` (`--ignore-missing-imports --show-error-codes`)
- **Pyright:** `pyright 1.1.414` (default config; `pyright positronixlab`)
- **Scope:** `positronixlab/` (117 source files)

## Overview

| Tool | Total Errors | Warnings | Non-Missing-Import Errors | Files Affected |
| :--- | ---: | ---: | ---: | ---: |
| `mypy` | 0 | 0 | 0 | 0 |
| `pyright` | 0 | 15 | 0 | 0 |

> **Note on Pyright `reportMissingImports` (197 hits):** Mostly optional extras (`ax`, `dash`, `plotly`, `cv2`, `umap`, `NeuroSurgeon`) and sibling `../parallelazy` when run in an isolated environment.

## Per-Module Breakdown

| Module | Mypy Errors | Pyright Errors (excl. `reportMissingImports`) |
| :--- | ---: | ---: |

## Top Mypy Error Codes

| Error Code | Count |
| :--- | ---: |

## Top Pyright Error Rules

| Rule | Count |
| :--- | ---: |

## Recommended `pyproject.toml` Additions

- Add `"ruff"`, `"mypy"`, `"pyright"` to `[project.optional-dependencies].dev`.
- Add `[tool.mypy]` (`python_version = "3.12"`, `ignore_missing_imports = true`, `exclude = ["positronixlab/baselines/lepori/_cvr_gen"]`).
- Add `[tool.pyright]` (`pythonVersion = "3.12"`, `typeCheckingMode = "basic"`, `exclude = ["positronixlab/baselines/lepori/_cvr_gen", "libs", "draft", "data"]`).

## Detailed Reports

- [`mypy_report.md`](mypy_report.md) — All mypy findings grouped by file.
- [`pyright_report.md`](pyright_report.md) — All non-`reportMissingImports` pyright findings grouped by file.
