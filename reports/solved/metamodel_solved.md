# `positronixlab/metamodel` — Solved Issues Archive

> **Archived:** 2026-11-27 (`b414936`).
> **Active open issues:** See [open/metamodel_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/open/metamodel_analysis.md).

---

## Resolved Bugs & Improvements

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **B2 (partial)** | Importing `MetaModelConfig`/`MetaTransform` pulled `parallelazy` | `metamodel/__init__.py` upgraded to PEP 562 lazy exports (also re-exports `ModelLoader`, `SOTAModel`, `OddOneOutHead`). *(Top-level `PipelineConfig` import in `metamodel.py`/`metacriterion.py` still open.)* |
| **B5 (`2ea821e`)** | Multi-layer CNN autoencoder crashed in `layer_size(ConvTranspose2d)` | Fixed during `2ea821e`; `test_cnn_autoencoder_multilayer_builds` passes without xfail. |
| **Cleanup** | `MetaTransform` lacked `__call__` & pickle-safe helpers; unused imports | Implemented `__call__` and top-level `_flatten`/`_to_float`/`_to_long` helpers; removed unused imports. |
| **B1 (P0, 2026-11-27)** | `MetaMetric.custom_metric` method shadowed by `self.custom_metric = {}` | Renamed dict to `self.custom_metrics` (`metacriterion.py:46, 59, 74-75`); xfail removed and passing. |
| **B3 (P0, 2026-11-27)** | `output_num_layers == 1` output `input_hidden_dim`; `output_hidden_dim` ignored | Checked `is_last_layer` before `is_first_layer` and used `output_hidden_dim` (`metamodel.py:252-261`); both xfails removed and passing. |
| **B8 (P0, 2026-11-27)** | `dropout_every == 0` enabled dropout on layer 0 instead of disabling | Removed `first_dropout = dropout_every == 0 and ...` across input/output/autoencoder blocks (`metamodel.py:225, 282, 326`); xfail removed and passing. |
