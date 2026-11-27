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
| **B5 (P1, 2026-11-27)** | Empty `DataLoader` crashed `process_epoch()` & `inference()` | Guarded `batch_cnt > 0` / `data_num > 0` and initialized `batch_idx = -1` (`deeplearning_base.py:119, 146`, `deeplearning_train.py:216-225`); xfail removed and passing. |
| **B6 (P1, 2026-11-27)** | `process_epoch()` double-updated `tqdm` progress bar | Removed duplicate `batch_bar.update()` (`deeplearning_base.py:135`). |
| **B7 (P1, 2026-11-27)** | Draft mode ran 21 batches instead of 20 | Changed `< 0` to `<= 0` (`deeplearning_base.py:140`, `deeplearning_train.py:206`); xfail removed and passing. |
| **B8 (P1, 2026-11-27)** | `DeeplearningTrain.on_batch_start()` dropped `with_data` | Added `with_data: bool = False` and forwarded to trainer/validator (`deeplearning_train.py:87-91`); xfail removed and passing. |
| **B9 (P1, 2026-11-27)** | Multiple `backward=True` metrics crashed without summing losses | Sums all `backward=True` losses before a single `.backward()` call and replaces `.data.item()` with `.detach().item()` (`deeplearning_base.py:87-105`); xfail removed and passing. |
