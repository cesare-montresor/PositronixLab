# PositronixLab v0.1.3 Code Review — Remaining Open Issues

> **Original Review:** 2026-11-26 against wheel `positronixlab-0.1.3-py3-none-any.whl`.
> **Status (2026-11-27 — `5740a0c` + working tree):** All P0/P1 confirmed bugs (**A1–A5, A6-batch, A7–A10, B1–B3, C1, C2-atlas, D1, E8**) have been **SOLVED** and archived in [solved/positronixlab_review_v0.1.3_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/positronixlab_review_v0.1.3_solved.md).
> Only **actual open items** remain below.

---

## 1. Open Summary Table

| ID | Sev. | Conf. | Location | Remaining Open Issue |
| :--- | :--- | :---: | :--- | :--- |
| **A6 (stream)** | Med | C | `modularcluster/modularcluster_kmeans_stream.py:77-81, 99-110, 265-268` | A6a (empty K range for `n∈{2,3}`) & A6b (silhouette with 1 populated cluster) fixed in `ModularClusterKmeans` only — **still open** in `ModularClusterKmeansStream` |
| **B4** | Med | V | `spacemath/scoring.py:19-20, 55-86, 139` | Sign-preserving scorers with ReLU; unconditional Softmax override even when `apply_activation=False`; `inf * 0 = NaN` mask multiply |
| **B5** | Low | V | `modularspace/custertracker.py:147, 156, 161, 165, 223` | Precision-only overlap; merge weighted by `old.sizes`; unbounded `history`; `assert old is not None` |
| **B6** | Low | L | `modularobserver/modularobserver.py:91-101` | `nodes_by_layer` keyed by module identity (collisions on shared modules); unused `module_ref` (`F841`) |
| **B7** | Low | C | `tasks/compositionality/atlas_check.py:199-225` | Legacy WIP `_cluster_output_attribution` shadows arguments & has unvalidated shape broadcast |
| **B8 (partial)** | Med | V | `dataprovider/dataprovider.py:420, 428`; `class_removal_dataprovider.py:102, 112`; `cvr_simclr_dataprovider.py:287` | `seed` defaults to `42`, but explicit `seed=None` falls back to global RNG; `ClassRemoval` & `CVRSimCLR` alias `dataset_test = dataset_val` |
| **C2–C10** | Low-Med | mixed | various | Robustness / API hazards (§2) |
| **D2–D5** | — | mixed | various | Performance opportunities (§3) |
| **E1–E7** | — | mixed | `pyproject.toml`, `README.md`, various | Packaging & dependencies (§4) |
| **F** | — | C | various | Code hygiene — 112 `ruff` hits incl. 6 new `F601` duplicate keys in `dataprovider/__init__.py:144-149` (§5) |

---

## 2. Open Robustness / API Hazards (C-series)

| ID | Location | Issue | Suggestion |
| :--- | :--- | :--- | :--- |
| **C3** | `tasks/atlas_discoverer.py:217-221, 310-311` | `model(x.to(device))` assumes single tensor input (dict/tuple batches break) | Pluggable `forward_fn`/`input_adapter` |
| **C5** | `tasks/atlas_discoverer.py:239-249` | Silent `k_max <= 8` cap and `n_units < 4` single-cluster fallback | Log fallback at `INFO`; expose cap |
| **C9** | `utils/utils.py:150-161` (`seed_everything`) | Sets `cudnn.benchmark = True`, uses `manual_seed` instead of `manual_seed_all`, no `deterministic` option | Add `deterministic: bool = False`, use `manual_seed_all` |
| **C10** | `utils/utils.py:338-356` (`safe_tensor`) | Broad `except Exception`; `"nan"` strings silently coerced via `float(val)` | Narrow exceptions; document accepted types |

---

## 3. Open Performance Opportunities (D-series)

| ID | Location | Observation | Suggestion |
| :--- | :--- | :--- | :--- |
| **D2** | `tasks/atlas_discoverer.py:542-565` | `discover*` runs two forward passes (`fit_atlas` + `_collect_mean_unit_energy`) | Collect energy during fit pass when loader is identical |
| **D3** | `modularcluster/modularcluster_kmeans.py:92-105, 233-235` | O(n²) CPU `distance_cache` for silhouette | Add `sample_size=` or threshold precompute |
| **D4** | `modularcluster/modularcluster_kmeans.py:129` | Extra `kmeans.assign(x_np)` after `train` | Minor; note transfer cost if GPU-backed |
| **D5** *(partial)* | `spacemath/common.py:82-106` | `POSITRONIXLAB_COMPILE=0` switch added, but still compiles on CPU by default | Skip `torch.compile` on CPU by default |

---

## 4. Open Packaging & Dependencies (E-series)

- **E1 — Broken README links on PyPI:** Relative links (`docs/`, `examples/tasks/example_atlas_vs_sota.py`) 404 on PyPI; switch to absolute GitHub URLs.
- **E2 — Heavy hard dependencies used in few files:** `torchsummary` (`metamodel/metamodel.py:11`), `pandas`/`requests` (`higgs_dataprovider.py`, `scan_dataprovider.py`, `math_dataprovider.py`) — make optional or lazy-import.
- **E3 — `faiss-cpu` only:** Add optional `gpu` extra or document choice.
- **E4 — `Requires-Python >=3.12`:** Verify whether 3.10/3.11 can be supported.
- **E5 — Optional-dependency top-level imports without extra hints:** `nas/experiment.py:29-31` (`ax.*`), `template/template_base.py:13-17` (`ax.*`), `modularui/*` (`dash`/`plotly`/`flask`), `baselines/lepori/models.py` (`transformers`), `_cvr_gen/{shape,utils}.py` (`cv2`, `PIL`). Raise `ImportError("pip install 'positronixlab[<extra>]'")`.
- **E6 — `py.typed` shipped without type-checker CI:** 13 implicit-`Optional` (`RUF013`) hits remain; either run `mypy`/`pyright` in CI or remove `py.typed`.
- **E7 — `torch>=2.0` lower bound:** Verify `nn.init.kaiming_uniform_(..., generator=gen)` (`tasks/masks.py:412`) minimum PyTorch version.

---

## 5. Open Code Hygiene (F)

112 `ruff` hits across `F,B,PLE,PLW,RUF,S,E7,E9,A,PERF`:
- **Duplicate dict keys (`F601`, 6 — NEW):** `dataprovider/__init__.py:144-149`.
- **One-line `if` (`E701`, 40):** `metamodel.py`, `utils.py`, `_cvr_gen/subnetwork_tasks.py`, `modularspace.py`, `modularobserver.py`, `metacriterion.py`.
- **Implicit `Optional` (`RUF013`, 13):** `deeplearning/`, `metamodel/`, `modularui/docks.py`, `template/`.
- **Unused locals (`F841`, 10):** `_cvr_gen/{shape,utils}.py`, `metamodel.py:199, 297, 301`, `modularobserver.py:97`.
- **`zip()` without `strict=` (`B905`, 9+):** `baselines/csordas/models.py`, `modularspace/`, `modularui/`, `tasks/compositionality/`, `dataprovider/`.
- **Star import & `print()` in library code:** `_cvr_gen/subnetwork_tasks.py:4` (`from ...utils import *`), ~60 `print()` calls across library modules.
- **Misc:** `metamodel.py:213` (`input` shadowing), `modularui.py:76` (`== False`), `stats/reducer.py:25` (`B027`), `class_removal_dataprovider.py:26, 29` (`B009`).
