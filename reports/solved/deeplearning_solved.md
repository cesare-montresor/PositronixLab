# `positronixlab/deeplearning` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [deeplearning_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/deeplearning_analysis.md).

---

## Resolved Bugs & Cleanup

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **B1** | `EarlyStopping.should_stop()` only maximized metrics, initialized `metric_last = 0`, off-by-one patience, broken `%` formatting | Fixed in `deeplearning_train.py:52-62, 230-263` (`minimize` auto-inferred via `"loss" in metric.lower()`, `metric_last=None`, `num_attempts >= max_attempts`, `{gain:+.4f}`). `test_b1_*` passes. |
| **B2** | `DeeplearningTrain.inference()` under-reported `avg_loss` by `~batch_size` | Fixed in `deeplearning_train.py:187` (`total_loss += loss.detach().item() * len(x)`). `test_b2_*` passes. |
| **B3** | `DeeplearningTrain.inference()` crashed on 3-tuple `(x, y, idx)` batches | Fixed in `deeplearning_train.py:176-180` (handles `len(batch) == 3`). `test_b3_*` passes. |
| **Cleanup** | Legacy file `deeplearning_legacy.py` in package tree | Removed from `positronixlab/deeplearning/`. |
| **B4 (P0, 2026-11-27)** | `DeeplearningBase.process_batch()` ran `self.model(x)` outside `torch.set_grad_enabled(self.is_training)` | Wrapped `self.model(x)` in `with torch.set_grad_enabled(self.is_training):` (`deeplearning_base.py:81-82`); xfail removed and passing. |
