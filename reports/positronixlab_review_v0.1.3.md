# PositronixLab v0.1.3 — Code Review

**Target:** `pip install positronixlab` → wheel `positronixlab-0.1.3-py3-none-any.whl` (≈21.7k lines of Python, 120 modules)
**Repo (from wheel metadata):** https://github.com/cesare-montresor/PositronixLab
**Date of review:** 2026-11-26

---

## 0. Scope, method, limits

**What I did**

- Downloaded the wheel from PyPI, unpacked it, and ran `pyflakes` and `ruff` (rule sets `F,B,PLE,PLW,RUF,S,E7,E9,A,PERF`) over `positronixlab/`.
- Read the core code paths in full or in large part: `tasks/masks.py`, `tasks/atlas_discoverer.py`, `spacemath/scoring.py`, `spacemath/common.py`, `modularcluster/modularcluster_kmeans.py`, `modularspace/custertracker.py`, `modularobserver/layerobserver.py` + `modularobserver.py`, `utils/utils.py`, `dataprovider/dataprovider.py` (split logic), parts of `pipeline/` and `nas/`.

**What I did *not* do**

- **Nothing was executed.** No torch install in my sandbox, so every runtime claim below comes from reading the code. Each finding carries a confidence tag.
- Not reviewed: vendored `baselines/lepori/_cvr_gen/*` (≈3.3k lines), the individual dataproviders (except split logic), `modularui/*` internals, `nas/*` internals, `baselines/*` adapters, `modularspace/space_step.py`, `spacemath/{contrast,sigreg,losses,weighting}.py`.
- The wheel does not ship `tests/`, `docs/` or `examples/`, so I could not check what is already covered.

**Line numbers** refer to the unpacked wheel `0.1.3`, relative to the `positronixlab/` package directory (e.g. `tasks/masks.py:367`). They may drift on `main`.

**Confidence tags**

- **[C] Confirmed** — the code as written does this; no assumptions about runtime data.
- **[L] Likely** — follows from the code under plausible inputs, but depends on something I did not read.
- **[V] Verify** — a methodological hypothesis; needs an experiment or test to settle.

---

## 1. Summary table

| ID | Sev. | Conf. | Location | Title |
|----|------|-------|----------|-------|
| A1 | Med | C | `__init__.py:6` | `__version__` reports **parallelazy**'s version |
| A2 | High | C | `tasks/masks.py:367` | `MaskApplier(seed=…)` not reproducible (`hash()` is salted) |
| A3 | High | C | `tasks/masks.py:349-406` | `MaskApplier` not exception-safe in `__enter__`; tied-weights restore bug |
| A4 | High | C | `tasks/atlas_discoverer.py:25-40` | `_reduce_activation_to_2d` wrong for Conv1d / Conv3d |
| A5 | Med | C | `modularobserver/layerobserver.py:38-39` | `on_grad_weights`: late-binding closure, `None` callback, leaked handles |
| A6 | Med | C | `modularcluster/modularcluster_kmeans.py:68-72, 195-241` | Empty K range for n∈{2,3}; silhouette crash on 1 populated cluster; `gpu=True` with faiss-cpu |
| A7 | Med | C | `utils/utils.py:33-43` | `suppress_output` is process-global and racy |
| A8 | Med | C | `utils/utils.py:20, 45-54` | `session_path()` returns the first path forever |
| A9 | Low-Med | C | `utils/utils.py:179-184, 217-246`; `pipeline/config.py:24` | Silent `None` returns, dropped kwargs, non-serializable hashing, md5 under FIPS |
| A10 | Med | C | `spacemath/common.py:80, 114-155` | `clear_torch_compile_cache`: dead docstring, risky `rmtree`, wrong user dir; `compile` shadows builtin |
| B1 | High | V | `tasks/masks.py:349-397` | Biases (and norm-layer affines) are never masked → "leak-free" claim |
| B2 | High | V | `tasks/atlas_discoverer.py:237-345, 420-441` | Subtask attribution is non-differential (mean \|act\| per cluster) |
| B3 | High | C | `tasks/atlas_discoverer.py:282, 302-303, 406-441` | Empty/one-shot dataloader → empty mask → silent no-op ablation |
| B4 | Med | V | `spacemath/scoring.py:55-67, 127-140` | Sign-preserving scorers with ReLU; silent Softmax override |
| B5 | Low | V | `modularspace/custertracker.py:147-165, 223` | Identity matching is precision-only; unbounded `history` |
| B6 | Low | L | `modularobserver/modularobserver.py:91-101` | `nodes_by_layer` keyed by module → collisions for shared modules |
| B7 | Low | C | `tasks/compositionality/atlas_check.py:199-222` | WIP function shadows arguments; shape logic suspect |
| B8 | Med | V | `dataprovider/dataprovider.py:195-203, 338, 419-445` | Split seed semantics: possible cross-process train/val inconsistency |
| C1–C10 | Low-Med | mixed | various | Robustness / API hazards (§4) |
| D1–D5 | — | mixed | various | Performance opportunities (§5) |
| E1–E8 | — | mixed | `pyproject`/metadata, various | Packaging & dependencies (§6) |
| F | — | C | various | Hygiene (§7) |

Recommended order: A1, A2, A5, A10-docstring (≤1 h total) → B3, A3, A4, A6 → B1, B2 experiments → everything else. See §9.

---

## 2. Confirmed bugs (A-series)

### A1 — `__version__` reads the wrong distribution  [C]

`__init__.py:3-8`

```python
from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("parallelazy")      # <-- line 6
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
```

`parallelazy` is a hard dependency, so it is always installed and `positronixlab.__version__` returns *its* version, never `0.1.3`. Anything that logs/records `positronixlab.__version__` (experiment provenance, NAS storage, bug reports) is recording the wrong thing.

**Fix:** `_package_version("positronixlab")`.
**Test:** `assert positronixlab.__version__ == importlib.metadata.version("positronixlab")`.

---

### A2 — `MaskApplier(seed=…)` is not reproducible across processes  [C]

`tasks/masks.py:365-367`

```python
if self.seed is not None:
    gen = torch.Generator(device=weight.device)
    gen.manual_seed(self.seed + (hash(layer_name) & 0xFFFF))      # line 367
```

`hash(str)` is randomized per interpreter start (`PYTHONHASHSEED`). The `seed` parameter therefore yields different random masks/weights for `random_ablate`, `randomly_sampled`, and `complement_sampled` in every process. For a benchmark whose purpose is "apples-to-apples" comparison (README), the random baselines are not reproducible, and results cannot be re-derived from a stored seed. Spawn-based workers (your `parallelazy` pipeline) are affected even within a single run, since each worker has its own salt.

Secondary: `& 0xFFFF` gives only 65 536 distinct offsets, so two layers can collide.

**Fix:**

```python
import zlib
layer_seed = (self.seed + zlib.crc32(layer_name.encode("utf-8"))) & 0x7FFFFFFF
gen.manual_seed(layer_seed)
```

**Test:** run a snippet in two subprocesses with different `PYTHONHASHSEED` and compare the resulting weights (see §8, T2).

---

### A3 — `MaskApplier` restoration guarantees are broken in two cases  [C]

`tasks/masks.py:312-324` (docstring), `349-397` (`__enter__`), `399-406` (`__exit__`)

The class docstring promises: *"Guarantees exact restoration of original weights even if an exception is raised."*

**A3a — exception inside `__enter__`.** Python does not call `__exit__` if `__enter__` raises. `__enter__` backs up and mutates layer by layer (`362`, `370-395`). If layer *k* fails (shape mismatch in `m_bool.to(weight.device)` at `363`, OOM in `torch.empty_like` at `374`, `_sample_random_binary_mask` error at `382-394`, a CUDA error), layers `0..k-1` stay modified and `self._backups` is never replayed. The model is silently left corrupted.

**A3b — tied / shared weights.** `__enter__` iterates `weight_mask.masks.items()`; if two names resolve to modules sharing the same `weight` tensor (tied embeddings, shared projection — the `[baselines]` extra pulls in `transformers`), the second `clone()` at `362` captures the *already masked* tensor. `__exit__` restores in insertion order (`402-405`), so the masked copy is written last and the weight stays permanently masked.

(`named_modules()` de-duplicates by module identity, but not by shared parameter tensors across *different* modules.)

**A3c — minor:** `weight.data.mul_(…)` (`370, 372, 385, 395`) bypasses autograd's version counter; if a graph was built before entering the context, a later `backward()` silently uses modified weights. Low priority.

**Fix sketch:**

```python
def __enter__(self):
    try:
        self._apply()
    except BaseException:
        self._restore()
        raise
    return self.model

def _apply(self):
    seen = set()
    ...
    for layer_name, m_bool in weight_mask.masks.items():
        ...
        ptr = weight.data_ptr()
        if ptr in seen:          # tied weights: first mask wins (document it) or intersect masks
            continue
        seen.add(ptr)
        self._backups[layer_name] = weight.data.detach().clone()
        ...

def _restore(self):
    modules = dict(self.model.named_modules())
    with torch.no_grad():
        for layer_name, orig_w in reversed(list(self._backups.items())):
            ...
    self._backups.clear()
```

**Tests:** T3a (inject a failing layer), T3b (tied-weight toy model) in §8.

---

### A4 — `_reduce_activation_to_2d` is wrong for Conv1d and Conv3d  [C]

`tasks/atlas_discoverer.py:25-40` (reducer), `43-47` (`_infer_out_units` explicitly supports `Conv1d/2d/3d`), call site `258`

```python
if act.ndim == 2: return act
if act.ndim == 4: return act.mean(dim=(2, 3))
if act.ndim == 3: return act.mean(dim=1)              # line 39
return act.reshape(act.shape[0], -1)
```

The docstring says Conv1d is "handled via mean over last dim", but the code averages `dim=1` for **every** 3-D tensor.

- Conv1d output `(B, C, L)` → returns `(B, L)`: it averages over **channels** and keeps length, so the energy vector has `L` entries instead of `out_channels`.
- Conv3d output `(B, C, D, H, W)` falls through to the flatten and returns `(B, C·D·H·W)`.
- A Linear on a sequence `(B, T, U)` is the case `dim=1` is correct for.

Downstream, `_clusters_to_unit_mask` (`286-345`) indexes `energy_cpu[c_mask]` where `c_mask` has length `n_units = out_channels` (`207`). If `L != C` this raises an `IndexError` (mask/tensor shape mismatch). If `L == C` by coincidence it silently produces a wrong mask.

**Fix:** dispatch on the module (or on a `channel_dim` argument), not on `ndim`:

```python
def _reduce_activation_to_2d(act, module=None):
    if act.ndim == 2:
        return act
    if isinstance(module, nn.modules.conv._ConvNd):
        return act.mean(dim=tuple(range(2, act.ndim)))     # (B, C, *spatial) -> (B, C)
    if act.ndim == 3:                                       # (B, T, U) sequence layers
        return act.mean(dim=1)
    return act.reshape(act.shape[0], -1)
```

Update every call site (`grep -n _reduce_activation_to_2d tasks/atlas_discoverer.py` — one is at `258`; the `fit_atlas` hook at `117-200` may have its own path, which I did not trace).

**Test:** T4 in §8 (Conv1d with C≠L).

---

### A5 — `LayerObserver.on_grad_weights`  [C]

`modularobserver/layerobserver.py:38-39`

```python
def on_grad_weights(self, callback: WeightCallback = None):
    [param.register_hook(lambda grad: callback(self.layer, param, grad)) for param in self.layer.parameters()]
```

Three defects:

1. **Late binding.** The lambda closes over the comprehension variable `param`; after the comprehension finishes, all hooks see the *last* parameter. (ruff B023.) A callback that logs per-parameter gradients receives the right `grad` but the wrong `param` for every hook but the last.
2. **`callback=None` is the default and is invoked.** Unlike `on_pre_forward_layer`/`on_forward_layer`/`on_backward_layer` (lines `41-60`), there is no `None` → "remove hook" branch, so a `None` callback raises `TypeError` at the first backward.
3. **Handles discarded.** `register_hook` handles are dropped, so the hooks can never be removed, and calling `on_grad_weights` twice stacks duplicate hooks. The `forward_weights_hook`/`backward_weights_hook` attributes (`34-35`) exist but are never used.

**Fix:**

```python
def on_grad_weights(self, callback=None):
    for h in self._weight_grad_handles:
        h.remove()
    self._weight_grad_handles = []
    if callback is None:
        return
    for param in self.layer.parameters():
        self._weight_grad_handles.append(
            param.register_hook(lambda grad, p=param: callback(self.layer, p, grad))
        )
```

(initialize `self._weight_grad_handles = []` in `__init__`).

---

### A6 — `ModularClusterKmeans`: three edge-case failures  [C]

`modularcluster/modularcluster_kmeans.py`

**A6a — empty K range for tiny n (`68-72`, `195-241`).**

```python
k_min = self.k_min or max(2, int(n ** 0.5 / 2))
k_max = self.k_max or min(int((2 * n) ** 0.5), n // 2)
return max(2, k_min), min(k_max, n - 1)
```

- n = 2: `k_max = min(2, 1) = 1` → returns `(2, 1)`.
- n = 3: `k_max = min(2, 1) = 1` → returns `(2, 1)`.
- Any explicit `k_min > n - 1` gives the same.

`_adaptive_search(x, 2, 1)` evaluates nothing (`range(2, 2)`), so `best_k`, `centroids`, `labels` stay `None` and `fit()` returns without error. `fit_predict()` (`243`) then returns `None`. With `verbose=True`, `self.silhouette_scores[self.best_k]` at `228` raises `KeyError(None)`.

The guard `if n < 2: raise` at `~201` is therefore insufficient. `AtlasDiscoverer` papers over this with `n_units < 4` (`atlas_discoverer.py:207`) and a `labels is None` fallback (`~224-229`), but any other caller is exposed.

**A6b — silhouette with one populated cluster (`90-99`).** `silhouette_score` raises `ValueError` ("Number of labels is 1") when FAISS returns fewer than 2 distinct labels (empty clusters, duplicate points — plausible for dead units whose embeddings collapse). The only guard is `if k < 2` (`~91`), which looks at the *requested* k, not the labels actually produced. No `try/except` in `_evaluate_k` (`129-148`).

**A6c — `gpu=True` with a CPU-only wheel (`119`).** `gpu=self.device.type == 'cuda'` is passed to `faiss.Kmeans`, but `faiss-cpu` is a hard dependency (`Requires-Dist: faiss-cpu`). With `faiss-cpu`, GPU symbols do not exist and `Kmeans.train` raises. Latent today because `AtlasDiscoverer` passes `torch.device("cpu")` (`atlas_discoverer.py:219-225`), but `ModularClusterKmeans(device=cuda_device)` and the streaming variant would break.

**Fixes:**

```python
# A6a: in fit(), after _compute_k_range
if k_min > k_max:
    self.best_k, self.best_score = 1, -1.0
    self.labels = torch.zeros(n, dtype=torch.long, device=self.device)
    self.centroids = x.mean(dim=0, keepdim=True)
    return self
    # or: raise ValueError(f"n={n} too small for K range [{k_min},{k_max}]")

# A6b
labels_np = labels.cpu().numpy()
if len(np.unique(labels_np)) < 2:
    return -1.0

# A6c
use_gpu = self.device.type == "cuda" and hasattr(faiss, "StandardGpuResources")
```

Also (A6d, trivial): `dists, labels = kmeans.assign(x_np)` at `124` — `dists` is unused (ruff RUF059).

**Tests:** T6 in §8.

---

### A7 — `suppress_output` is a process-global, racy fd redirect  [C]

`utils/utils.py:33-43`

```python
def suppress_output(stderr: bool = True):
    stderr_fd = 2 if stderr else 1
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    try: yield
    finally:
        os.dup2(old_stderr, stderr_fd); os.close(devnull); os.close(old_stderr)
```

Used at `modularcluster/modularcluster_kmeans.py:111` (and in the streaming variant).

1. **Race.** Two threads entering concurrently: thread B's `os.dup(2)` can capture the already-redirected fd (→ `/dev/null`). When A exits it restores the real stderr, then B exits and restores `/dev/null`. stderr stays silenced for the rest of the process. The package advertises a "thread-safe Dash/Plotly" dashboard, so a UI thread and a clustering thread in one process is a plausible scenario.
2. **Collateral suppression.** Any thread that writes to fd 2 during the context loses its output (Dash/Flask logs, tqdm, tracebacks from other threads).
3. **Python buffers aren't flushed first**, so buffered text written before entry may appear after exit or be dropped.
4. **Naming.** `stderr=False` suppresses **stdout** (fd 1). The parameter is really a stream selector.

Since the only consumer passes `verbose=self.verbose` to FAISS already (`~116`), consider deleting the wrapper. If you keep it, use a lock + depth counter:

```python
_SUPPRESS_LOCK = threading.RLock()
_SUPPRESS_STATE: dict[int, tuple[int, int, int]] = {}   # fd -> (depth, saved_fd, devnull_fd)

@contextmanager
def suppress_output(stderr: bool = True):
    fd = 2 if stderr else 1
    with _SUPPRESS_LOCK:
        depth, saved, null = _SUPPRESS_STATE.get(fd, (0, -1, -1))
        if depth == 0:
            (sys.stderr if fd == 2 else sys.stdout).flush()
            saved, null = os.dup(fd), os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, fd)
        _SUPPRESS_STATE[fd] = (depth + 1, saved, null)
    try:
        yield
    finally:
        with _SUPPRESS_LOCK:
            depth, saved, null = _SUPPRESS_STATE[fd]
            if depth == 1:
                os.dup2(saved, fd); os.close(saved); os.close(null)
                del _SUPPRESS_STATE[fd]
            else:
                _SUPPRESS_STATE[fd] = (depth - 1, saved, null)
```

(Point 2 is inherent to fd-level redirection.)

---

### A8 — `session_path()` ignores its arguments after the first call  [C]

`utils/utils.py:20, 45-54`

```python
last_session_path = None                                              # 20
def session_path(name, uid=None, createDirectory=True, force=False):  # 45
    global last_session_path
    if last_session_path is not None and not force: return last_session_path   # 47
    basedir = 'output/'
    ...
    last_session_path = path                                          # 52
```

The first call wins for the lifetime of the process. A second experiment calling `session_path("expB")` silently writes into `output/expA_<ts>/`. Also: `basedir = 'output/'` is relative to the CWD (so results land in different places depending on where the script is launched), paths are built with `+`/`'/'` instead of `os.path.join`/`pathlib`, and `name` is not sanitized.

**Fix:** cache per `(name, uid)` (a dict), or drop the global and let callers hold the returned path; make `basedir` configurable (env var or argument) and use `pathlib.Path`.

---

### A9 — Serialization / hashing helpers  [C]

**A9a — `load_data` returns `None` for two different failures.** `utils/utils.py:217-246`: missing file (`227`) and unsupported extension (`246`) both return `None`, which is indistinguishable from a legitimately-null JSON/YAML value. Callers that do `cfg = load_data(p); cfg["x"]` fail later with `TypeError: 'NoneType'…` far from the cause. Prefer `FileNotFoundError` / `ValueError`, or an explicit `default=` parameter.

**A9b — YAML kwargs silently dropped.** `236-237`: `yaml.load(f, **kwargs)` only if `'Loader' in kwargs`; otherwise `yaml.safe_load(f)` and all other kwargs are discarded without warning. Raise on unexpected kwargs.

**A9c — `signature_settings` is brittle.** `179-184`: `json.dumps(data, sort_keys=True)` has no `default=`, so `Path`, numpy scalars, `torch.dtype`, dataclasses raise `TypeError`. A naive `default=str` makes the hash unstable for objects whose `repr` embeds an address — a custom encoder (Path→str, np scalar→item, dtype→str, dataclass→asdict) is safer. `hashlib.new("md5", …)` raises `ValueError` on FIPS-enabled systems; pass `usedforsecurity=False`.

**A9d — same md5 in `pipeline/config.py:24`:** `hashlib.md5(content.encode("utf-8")).hexdigest()` → `hashlib.md5(..., usedforsecurity=False)`.

**A9e — `pickle` paths.** `utils/utils.py:240` (generic `.pkl` branch of `load_data`) and `dataprovider/syneval_oddoneout_dataprovider.py:92` (`pickle.load(f)[:n_samples]`). Fine for trusted bundled data; add a docstring warning so nobody points `load_data` at downloaded files.

---

### A10 — `spacemath/common.py`: compile utilities  [C]

**A10a — dead docstring.** `common.py:114-119`: inside `clear_torch_compile_cache`, the `if verbose: print(...)` at `116-117` comes *before* the string at `119`, so the docstring is just an expression statement; `help(clear_torch_compile_cache)` shows nothing.

**A10b — destructive deletes of user-controlled paths.** `common.py:135-146`: the function `rmtree`s `/tmp/torchinductor_$USER`, `$TORCHINDUCTOR_CACHE_DIR`, `~/.triton/cache`, `$TRITON_CACHE_DIR`. Environment variables can legitimately point at shared or parent directories (the README itself suggests setting them); `rmtree` on the wrong value is data loss. And wiping `~/.triton/cache` invalidates caches of unrelated concurrent jobs. Guard with a name check (`torchinductor_*`, `triton*`), refuse paths that are `$HOME`, `/tmp`, `/`, or make deletion opt-in per directory.

**A10c — wrong user directory.** `common.py:135` uses `os.environ.get('USER', '')`. Torch derives its default cache dir from `getpass.getuser()`; `$USER` is often unset in containers/cron/Windows, producing `/tmp/torchinductor_` (also flagged by ruff S108). Ask torch for the directory (check `torch._inductor.config` / `torch._inductor.codecache` for your supported versions) instead of reconstructing it.

**A10d — builtin shadowing.** `common.py:80`: `def compile(func)` shadows the builtin `compile`. Any module doing `from positronixlab.spacemath.common import *` (or defining the name at module level) loses the builtin. Rename to `torch_compile`.

**A10e — no way to disable compilation.** `common.py:13` (`default_compile_mode = "default"`), `92` (`torch.compile(func, mode=default_compile_mode)`). There is no env switch or API to turn compilation off (debugging, platforms where `torch.compile` is unsupported, read-only `$HOME`). See D5.

---

## 3. Methodological concerns (B-series)

These are not crashes. They can change the scientific conclusions, so I'd settle them with experiments before trusting benchmark numbers.

### B1 — Biases and norm-layer affines are not masked  [V]

`tasks/masks.py:349-397`; `grep -n bias tasks/masks.py` → **0 hits**.

`MaskApplier.__enter__` only touches `mod.weight` (`getattr(mod, "weight", None)` at `358`). Consequence for unit-granularity masks (which `AtlasDiscoverer` produces, `atlas_discoverer.py:340-345`, `granularity="unit"`):

- In `retain` mode, an ablated unit has an all-zero weight row but keeps its bias → pre-activation = `bias`, post-activation = `act(bias)`, which is generally ≠ 0 and flows into the next layer. The "retained subnetwork" therefore still receives signal from ablated units.
- In `zero_ablate` the situation is the same in reverse.
- If a `BatchNorm`/`LayerNorm` follows a masked Linear/Conv, an ablated unit's zero output is shifted by `β − μγ/σ`, which is also non-zero. (I did not check how `MASKABLE_LAYER_TYPES` in `baselines/base.py:28` treats norm layers.)

The class docstring calls itself "Leak-free". Test models with `bias=False` (some Csordás-style baselines) would hide the problem; models with bias would not.

**Suggested handling:** add `bias_mode: Literal["keep", "mask"]`; when the source mask is unit-level, zero `bias[unit]` for ablated units (and document what happens with norm layers). Keep `"keep"` available to reproduce published weight-only baselines.

**Experiment (T1):** Linear(4→3) with bias, unit 0 ablated in `retain`, assert the layer's output for unit 0 is exactly 0.

---

### B2 — Subtask attribution is not differential  [V]

`tasks/atlas_discoverer.py:237-283` (`_collect_mean_unit_energy`), `286-345` (`_clusters_to_unit_mask`), `420-441` (`discover_subtasks`), selection at `314-319`.

For each subtask, the score of a cluster is the **mean absolute activation of its units on that subtask's loader** (`258`: `…abs().cpu()`; `306-307`: `energy_cpu[c_mask].mean()`), and the top `max(min_active_clusters, round(n_clusters · frac))` clusters are kept. The reference loader is only used to *fit* the atlas; its statistics are not used to normalize the subtask energies.

Why this is likely a problem:

- Units/clusters with large typical magnitudes rank first for **every** subtask, so masks for "add" and "mul" probably overlap heavily — the property the "one unsupervised pass → masks for both subtasks" claim needs to avoid.
- With few clusters (README uses `k_min=2, k_max=4`; default cap is 8 at `212`), "top fraction" picks one or two clusters; identical picks across subtasks are the most likely outcome.
- Hooks are registered on the target layer modules (`mod.register_forward_hook`, `~255`). If those are `Linear`/`Conv` modules, the hook sees **pre-activations**, and `.abs()` counts strongly *negative* (i.e., switched-off after ReLU) units as high energy.

**Suggested experiments / changes:**

1. Log pairwise Jaccard/IoU between subtask masks per layer in the mask metadata (cheap, reveals collapse immediately).
2. Score by *selectivity*: `(E_sub − E_ref) / (E_ref + ε)` or a z-score against the reference loader; or contrast against the *other* subtasks.
3. Try signed/post-activation energy (clamp at 0 for ReLU nets) and compare.
4. `tasks/compositionality/atlas_check.py` already computes cluster × class statistics (`~170-195`); a cluster-class selectivity score might be a better attribution signal than energy.

---

### B3 — Empty energy dict → empty mask → silent no-op ablation  [C]

`tasks/atlas_discoverer.py:282` (return `{k: … for k in target_layers if k in sums}`), `302-303` (`if energy is None: continue`), `406-418` (`discover`), `420-441` (`discover_subtasks`), `MaskApplier.__enter__` at `masks.py:354`.

`dataloader` is typed `Iterable[Any]`. `discover()` iterates it twice: once in `fit_atlas`, once in `_collect_mean_unit_energy`. A generator / one-shot iterator is exhausted after the first pass, so the energy pass sees zero batches → `sums` is empty → `unit_energy` is `{}` → every layer hits `continue` at `302-303` → a `SubnetworkMask` with **no layers**. `MaskApplier` iterates `weight_mask.masks.items()` (empty) and changes nothing, so the *full model* gets evaluated under an "atlas:*" mask name. The same silent path exists for an empty loader or a layer the hook never fired for.

**Fix:** after `_collect_mean_unit_energy`, check `counts[name] > 0` for every target layer and raise (or at least `logger.warning`) otherwise; in `_clusters_to_unit_mask` raise if `masks` ends up empty. Optionally reject non-re-iterable inputs up front (`iter(x) is x`).

**Test:** T3 in §8.

---

### B4 — Sign-preserving scorers and activation choice  [V]

`spacemath/scoring.py:55-67` (`score_sign_preserving_row`), `70-82` (`_batch`), `112-140` (`activations_to_scores`), `127-129` (`apply_activation`), `139-140` (Softmax override)

- `score_sign_preserving_*` map positives to `[0, 1]` and negatives to `[-1, 0)`. If `apply_activation=True` and `activation_fn` is a ReLU-family function, the negative branch is identically zero and the output range collapses to `[0, 1]`: the "off" side of the contrastive objective (`idxs_off`) has no negative mass. Consider a warning when a sign-preserving scorer is combined with a non-negative activation.
- `139`: `effective_method = "softmax" if isinstance(activation_fn, nn.Softmax) else score_method` overrides the user's `score_method` whenever the *argument* is a Softmax, even when `apply_activation=False` and the Softmax is never applied. Make the override conditional on `apply_activation` or at least log it.
- `score_normalize_by_threshold` (`10-23`): `(values − threshold + eps) * idxs_on` propagates NaN/inf (`inf * 0 = NaN`); use `torch.where` if inputs can be non-finite.
- Row-wise normalization (`amax(dim=1)`) is relative to the sample; batch-wise (`amax()`) couples samples within a batch. This matters for reproducibility across batch sizes; worth a note in the docs.

---

### B5 — `ClusterTracker`  [V / minor]

`modularspace/custertracker.py`

- `165`: `frac = overlap / sizes_new` — overlap is normalized by the **new** cluster size (precision), never by the old one (recall). A 10-unit fragment of a 100-unit old cluster is a perfect (frac = 1.0) "match" and inherits the old identity; the other 90 units are only recovered as further "split" children. Probably intentional, but there is no test or comment showing it is. Consider a symmetric criterion or documenting this.
- `223`: merged position is weighted by `old.sizes[...]`, not by the overlap, so a minor contributor with a big old cluster dominates.
- `147`, `156`: `self.history.append(events)` — unbounded growth over long training runs. Use `collections.deque(maxlen=N)` or make history optional.
- `161`: `assert old is not None` → explicit exception (asserts vanish under `python -O`).
- `_one_hot` (`~58-66`) builds a dense `[N, K]` float32 tensor each step; fine at current scale (K small) but worth noting if N reaches 1e5+.

(The generator device handling at `123` and `86-87` is consistent — I checked it for the CUDA-generator/CPU-tensor mismatch and it is fine.)

---

### B6 — `ModularObserver` keyed by module identity  [L]

`modularobserver/modularobserver.py:91-101`

```python
for name, node in self.mapper.graph.nodes.items():
    if not self._is_observable_layer(node.module_ref, name): continue
    if node.name not in self.nodes_by_name:
        module_ref = node.module_ref                     # 97: assigned, never used
        self.nodes_list.append(node)
        self.nodes_by_name[node.name] = node
        self.nodes_by_layer[node.module_ref] = node      # 100
```

If one `nn.Module` instance appears at several graph nodes (a reused activation, weight sharing, a recurrent cell called per step), `nodes_by_layer` keeps only the last node and `by_layer(layer)` (`~108`) returns that one. `97` is dead code (F841). Also `modelmapper.py:137` defines `__hash__` via a private `__key()`; verify `__eq__` is consistent with it.

---

### B7 — `_cluster_output_attribution` (WIP)  [C]

`tasks/compositionality/atlas_check.py:199-222`

- The docstring says "WIP".
- `217`: `for (cluster,y_hats),counts in zip(unique, counts):` rebinds the method argument `y_hats` and the tensor `counts` as loop targets. It works today only because `zip(unique, counts)` is evaluated once before rebinding; it will confuse every future reader (ruff PLW2901, B020).
- `210`: `ys_on = torch.cat([(y_hats * idx_on)[idx_on] for idx_on in idxs_on])` — `idx_on` is a per-sample boolean over features, while `y_hats` was unsqueezed at `203`; I did not trace shapes end-to-end, but indexing a tensor's batch dimension with a feature-length mask only lines up when `bs == num_features`. Treat as unvalidated until there is a test.

Recommendation: don't export it from the package until it has a test.

---

### B8 — Split-seed semantics  [V]

`dataprovider/dataprovider.py:195-203` (`BaseTensorDataset.random_split(…, seed=42)`), `338` (`default_split = (0.7, 0.3)`), `378-400`, `419-445` (`_split_dataset`)

- `_split_dataset` uses `use_seed = self.seed if seed is None else seed`. If a provider is built with `seed=None`, `torch.Generator` is `None` → `random_split` draws from the global RNG. Two processes (e.g., a training worker and an evaluation worker, both spawned by the `parallelazy` pipeline) that each construct the provider would get **different** train/val partitions → validation samples of one are training samples of the other → silent leakage. I did not check the default of `self.seed` — verify.
- `BaseTensorDataset.random_split` defaults to `seed=42`, while `_split_dataset` passes whatever `use_seed` is (possibly `None`), so the two entry points behave differently.
- `default_split = (0.7, 0.3)` has no test split; `dataset_test` becomes `test_dataset` (possibly `None`, `445-448`). Check what `test_loader()` does in that case — evaluating on validation data under the name "test" would be a quiet methodological error.
- `HIGGSDataProvider` (`dataprovider/higgs_dataprovider.py:101-125`) has explicit `test_seed`/`split_seed`/`base_seed`, which is the right pattern to copy.

---

## 4. Robustness / API hazards (C-series)

| ID | Location | Issue | Suggestion |
|----|----------|-------|-----------|
| C1 | `tasks/masks.py:370, 372, 385, 395` | `weight.data.mul_(mask)`: `inf*0` and `NaN*0` give NaN | `torch.where(m, w, 0)` / `masked_fill_` |
| C2 | `tasks/atlas_discoverer.py:155, 245` | `next(model.parameters())` → `StopIteration` for parameterless models | `next(model.parameters(), None)` + fallback to CPU |
| C3 | `tasks/atlas_discoverer.py:276` | `model(x.to(device))` assumes a single tensor input (dict/tuple batches break) | handle via a pluggable `forward_fn`/`input_adapter` |
| C4 | `tasks/atlas_discoverer.py:230` | Comment cites a "float32 label bug" that no longer exists (`modularcluster_kmeans.py:125` already casts to `torch.long`) | Remove comment, keep the `.long()` as a cheap guard or drop it |
| C5 | `tasks/atlas_discoverer.py:207-215` | Hard cap `k_max ≤ 8` by default and `n_units < 4` → single cluster (whole layer always retained) are silent design choices | Log at INFO which layers fell back to one cluster; expose the cap |
| C6 | `nas/experiment.py:46`; `pipeline/pipeline.py:105, 186` | `except Exception: pass` (ruff S110) | At least `logger.debug(..., exc_info=True)` |
| C7 | `nas/experiment.py:338, 419`; `modularspace/custertracker.py:161` | `assert` used for runtime invariants (removed with `python -O`) | explicit `if … raise` |
| C8 | `dataprovider/higgs_dataprovider.py:403` | `warnings.warn` without `stacklevel` (B028) | `stacklevel=2` |
| C9 | `utils/utils.py:100-106` (`set_seed`) | Seeds `random`, `numpy`, `torch`, `torch.cuda.manual_seed` (current device only; `torch.manual_seed` already covers all devices) but not `cudnn.deterministic/benchmark`, nor `PYTHONHASHSEED` (which also feeds A2) | add `deterministic: bool = False` option; use `manual_seed_all` |
| C10 | `utils/utils.py:248-267` (`safe_tensor`) | Falls back through three coercions with a bare `except Exception`; strings like `"nan"` become tensors silently via `float(val)` | narrow the exceptions; document accepted types |

---

## 5. Performance opportunities (D-series)

| ID | Location | Observation | Suggestion |
|----|----------|-------------|-----------|
| D1 | `tasks/atlas_discoverer.py:258-263` | Per batch, per layer: `.abs().cpu()` forces a device→host sync, then CPU accumulation | accumulate on-device (`sums[name] += act2d.abs().sum(0)`), `.cpu()` once at the end |
| D2 | `tasks/atlas_discoverer.py:406-441` | `discover*` runs one forward pass for `fit_atlas` and a second one per loader for energy | collect energy during the fit pass when the loader is the same; or cache by loader id |
| D3 | `modularcluster/modularcluster_kmeans.py:90-99, 195-241` | Silhouette on CPU with a full `n×n` `distance_cache` (`precompute_distances=True` default) → O(n²) time and memory per K | `sample_size=`/`random_state=` in `silhouette_score`, or GPU silhouette; make precompute optional above a threshold |
| D4 | `modularcluster/modularcluster_kmeans.py:124` | Extra `kmeans.assign(x_np)` after `train` just to get labels | fine for now; if FAISS is GPU-backed this is another transfer |
| D5 | `spacemath/common.py:80-95` | Every decorated helper is `torch.compile`d at first use; for tiny CPU layers (Atlas default, small MLPs) compile latency can dominate | env/API switch (`POSITRONIXLAB_COMPILE=0`), and don't compile on CPU by default |

---

## 6. Packaging & dependencies (E-series)

**E1 — Broken README links on PyPI.** The README links `docs/` and `examples/tasks/example_atlas_vs_sota.py` relatively; neither is in the wheel (`RECORD` lists only `positronixlab/…`). On the PyPI project page those links 404. Use absolute `https://github.com/cesare-montresor/PositronixLab/blob/main/...` URLs.

**E2 — Heavy hard dependencies used in one file each.**

| Dependency | Only used in | Suggestion |
|------------|--------------|-----------|
| `torchsummary` (unmaintained) | `metamodel/metamodel.py:11` | replace with `torchinfo`, or lazy-import |
| `pandas`, `requests` | `dataprovider/higgs_dataprovider.py:12-13` (downloader, `requests.get` at `~334`) | `[higgs]` extra or lazy import |
| `scipy` | `modularcluster/streaming_cluster_tracker.py:6` (`linear_sum_assignment`) | keep (reasonable), or lazy |

**E3 — `faiss-cpu` as the only FAISS option.** See A6c. Add a `gpu` extra (`faiss-gpu`) or detect at runtime; document the choice.

**E4 — `Requires-Python >=3.12`.** A grep found no PEP 695 generics, `typing.override`, `itertools.batched` or `except*` (I did **not** check PEP 701 f-string nesting, and `ax-platform>=1.3` / your torch floor may impose their own minimum). If nothing needs 3.12, relaxing to 3.10/3.11 widens the audience (many clusters are still on 3.10/3.11). Classifiers list only 3.12 and 3.13.

**E5 — Optional-dependency imports at module top level.**

- `nas/experiment.py:25-27` imports `ax.*` at the top.
- `modularui/*` imports `dash`, `dash_bootstrap_components`, `plotly`, `dash_cytoscape`, `flask` at top level (`components.py:5-7`, `layouts.py:3-5`, `plots.py:3-8`, `settings.py:1`, `docks.py:1`, `modularui.py:10-11`).
- `baselines/lepori/_cvr_gen/shape.py:4` (`import cv2`), `_cvr_gen/utils.py:4` (`from PIL import Image`).

Package `__init__` files use lazy exports (good), but importing a submodule directly without its extra gives a bare `ModuleNotFoundError`. Wrap in `try/except ImportError` that raises `ImportError("pip install 'positronixlab[nas]'")`.

**E6 — `py.typed` is shipped** but there are 13 implicit-`Optional` annotations (RUF013) and no evidence of a type-checker run. Either run `mypy`/`pyright` in CI or drop the marker so downstream users don't get false guarantees.

**E7 — Lower bound of `torch>=2.0`.** `nn.init.kaiming_uniform_(…, generator=gen)` (`tasks/masks.py:376`) uses a `generator` kwarg that, as far as I recall, was added after 2.0 (I believe 2.1). **Unverified** — check the lowest torch you claim to support.

**E8 — Version string** — see A1.

---

## 7. Code hygiene (F)

From `ruff` (selected rules; 144 hits total after ignoring `F401/F403/F405`, star-import noise and Unicode in docstrings):

- **Corrections to my earlier summary:** I previously said "55 unused imports". 53 of those are in `dataprovider/__init__.py` (lazy re-export scaffolding), so they are not real. Real ones: `baselines/base.py` (1) and `baselines/lepori/_cvr_gen/__init__.py` (1, already `noqa`).
- **Unused local variables (F841, 10):** `baselines/lepori/_cvr_gen/shape.py:76, 143, 144, 338, 339`; `_cvr_gen/utils.py:331` (`angle`); `metamodel/metamodel.py:199, 297, 301` (`is_last_layer`, `is_first_layer`, `last_size`); `modularobserver/modularobserver.py:97`.
- **One-line `if x: y` (E701, 42):** `metamodel/metamodel.py` (17), `utils/utils.py` (13), `_cvr_gen/subnetwork_tasks.py` (6), plus single hits in `modularspace/modularspace.py:82`, `modularobserver/modularobserver.py:88`, `metamodel/metacriterion.py:21`.
- **Implicit Optional (RUF013, 13):** `deeplearning/deeplearning_base.py:185`, `deeplearning_train.py:57, 157`, `metamodel/metacriterion.py:42`, `metamodel/metamodel.py:17`, `modularui/docks.py:103`, `template/template_{covertype,fmnist,higgs,mnist,factory}.py`.
- **`zip()` without `strict=` (B905, 9):** `baselines/csordas/models.py:86`, `modularspace/custertracker.py:210`, `modularspace/modularspace.py:97`, `modularui/docks.py:113`, `modularui/modularui.py:281`, `tasks/compositionality/atlas_check.py:49, 191, 209, 217`. You're on 3.12 — `strict=True` turns silent truncation on length mismatch into an error.
- **Unused loop vars (B007, 7):** e.g. `deeplearning/deeplearning_base.py:137`, `baselines/neurosurgeon/adapter.py:311`, `tasks/compositionality/weightmask_check.py:149` (also PERF102: iterate `.values()`).
- **Builtin shadowing (A001):** `metamodel/metamodel.py:213` (`input`), `spacemath/common.py:80` (`compile`, A10d).
- **f-strings without placeholders (F541, 5):** `spacemath/common.py:117, 151`, `deeplearning/deeplearning_train.py:260`.
- **Star import in vendored code:** `baselines/lepori/_cvr_gen/subnetwork_tasks.py:4` (`from ...utils import *`) triggers ~70 "may be undefined" warnings in a 2 436-line file; replace with an explicit import list.
- **`print()` in library code (61 occurrences)**: `_cvr_gen/subnetwork_tasks.py` (13), `modularcluster/modularcluster_kmeans_stream.py` (11), `modularcluster/modularcluster_kmeans.py` (10), `utils/utils.py` (8), `spacemath/common.py` (5), `dataprovider/higgs_dataprovider.py` (5), `deeplearning/*` (5 total), `metamodel/metamodel.py` (2), `spacemath/spacemath.py` (2). Use `logging` (you already have `logger` in `utils.py`).
- **Misc:** `modularui/modularui.py:76` `== False`; `stats/reducer.py:25` empty method in ABC without `@abstractmethod` (B027); `modularui/layouts.py:51` unnecessary lambda; `dataprovider/dataprovider.py:208-209`, `spacemath/scoring.py:49`, `tasks/atlas_discoverer.py:316` redundant `int(int)` casts; `dataprovider/class_removal_dataprovider.py:26, 29` `getattr` with constant name (B009).
- **`__all__ = sorted(_LAZY_EXPORTS.keys())`** (`dataprovider/__init__.py:151`, `metamodel/__init__.py:20`, `modularui/__init__.py:27`): ruff flags PLE0605 but this is a valid runtime expression. Harmless; `from pkg import *` will trigger every lazy import.
- **Zero hits** for mutable default arguments (B006), bare `except:`, `eval/exec`, `yaml.load` without `Loader` in default paths (safe_load is the default).

---

## 8. Suggested regression tests

Pseudocode-level; adapt to your fixtures.

```python
# T0 — A1
def test_version_matches_distribution():
    import importlib.metadata as md, positronixlab
    assert positronixlab.__version__ == md.version("positronixlab")

# T1 — B1 (bias leak)
def test_retain_mask_zeroes_ablated_unit_output():
    lin = nn.Linear(4, 3, bias=True); nn.init.constant_(lin.bias, 1.0)
    mask = unit_mask({"": torch.tensor([True, False, True])})   # ablate unit 1
    with MaskApplier(lin, mask, mode="retain"):
        y = lin(torch.randn(8, 4))
    assert torch.all(y[:, 1] == 0)          # fails today: equals bias

# T2 — A2 (seed reproducibility across processes)
def _run(hashseed):
    code = "...build model, MaskApplier(seed=0, mode='random_ablate'), print(weight.sum().item())"
    return subprocess.check_output([sys.executable, "-c", code],
                                   env={**os.environ, "PYTHONHASHSEED": hashseed})
def test_seed_is_reproducible():
    assert _run("1") == _run("2")

# T3a — A3a (exception in __enter__ restores)
def test_enter_failure_restores_weights(monkeypatch):
    model = make_two_layer_model(); before = snapshot(model)
    monkeypatch.setattr(masks_mod, "_sample_random_binary_mask", boom_on_second_call)
    with pytest.raises(RuntimeError):
        with MaskApplier(model, some_mask, mode="randomly_sampled"): pass
    assert_equal(snapshot(model), before)

# T3b — A3b (tied weights)
def test_tied_weights_restored():
    a = nn.Linear(4, 4, bias=False); b = nn.Linear(4, 4, bias=False); b.weight = a.weight
    ...
    with MaskApplier(model, mask, mode="retain"): pass
    assert_equal(a.weight, original)

# T3 — B3 (empty energy)
def test_atlas_rejects_exhausted_loader():
    gen = (b for b in loader)      # one-shot
    with pytest.raises(ValueError):
        AtlasDiscoverer(...).discover(model, gen)

# T4 — A4 (Conv1d)
def test_reduce_conv1d_returns_channels():
    conv = nn.Conv1d(3, 5, 3); out = conv(torch.randn(2, 3, 16))   # (2, 5, 14)
    assert _reduce_activation_to_2d(out, module=conv).shape == (2, 5)

# T5 — A5 (late binding)
def test_grad_weights_hook_gets_right_param():
    seen = []; obs.on_grad_weights(lambda layer, p, g: seen.append(p))
    layer(x).sum().backward()
    assert {id(p) for p in seen} == {id(p) for p in layer.parameters()}

# T6 — A6
@pytest.mark.parametrize("n", [2, 3])
def test_kmeans_tiny_n_does_not_return_none(n):
    km = ModularClusterKmeans(k_min=2, k_max=4).fit(torch.randn(n, 4))
    assert km.labels is not None
def test_kmeans_duplicate_points_no_crash():
    ModularClusterKmeans().fit(torch.zeros(20, 4))      # one populated cluster

# T7 — B2 diagnostic (not a pass/fail test; log it)
def test_subtask_masks_not_identical():
    masks = atlas.discover_subtasks(model, ref, {"add": add, "mul": mul})
    for layer in masks["add"].masks:
        iou = jaccard(masks["add"].masks[layer], masks["mul"].masks[layer])
        assert iou < 1.0, f"{layer}: add and mul masks are identical"
```

---

## 9. Suggested fix order

| Priority | Items | Effort | Why |
|----------|-------|--------|-----|
| **P0 (≈1 h)** | A1, A2, A5, A10a (move docstring), A10d (rename `compile`) | S | trivial, high value for reproducibility/provenance |
| **P1** | B3 (empty-mask guard), A3 (exception safety + tied weights), A4 (conv reduce), A6 (kmeans edge cases), A8 (session_path), A9a/b (loader errors) | S–M | prevents silent wrong results |
| **P2 (experiments)** | B1 (bias leak), B2 (differential attribution + IoU logging), B8 (split seeds), B4 (sign-preserving/activation) | M | can change published conclusions; do before further benchmarking |
| **P3** | A7 (suppress_output), A10b/c (cache clear), A9c/d/e, C1–C10 | S–M | robustness |
| **P4** | D1–D5, E1–E7, F | M | performance, packaging, hygiene |

---

## Appendix — commands used

```bash
pip download positronixlab --no-deps -d dl          # wheel 0.1.3
unzip dl/positronixlab-0.1.3-py3-none-any.whl -d dl/x
python -m pyflakes positronixlab
ruff check positronixlab --select F,B,PLE,PLW,RUF,S,E7,E9,A,PERF \
     --ignore F403,F405,F401,RUF001,RUF002,RUF003,S101,E741,RUF012 --statistics
grep -rn "hash(\|bias\|manual_seed\|print(\|assert " positronixlab/
```

Nothing was run against torch. Any "likely"/"verify" item above should be treated as a hypothesis until a test confirms it.
