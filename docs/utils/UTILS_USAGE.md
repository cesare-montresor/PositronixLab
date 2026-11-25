# Utils Usage Guide

Because `positronixlab/utils/__init__.py` is currently empty, import functions from `positronixlab.utils.utils` (or import the submodule as `from positronixlab.utils import utils`):

```python
from positronixlab.utils import utils
from positronixlab.utils.utils import (
    assert_tensor,
    camel_to_kebab,
    camel_to_snake,
    f1s_from_truth_table,
    is_valid_tensor,
    load_data,
    model_hash,
    safe_tensor,
    save_data,
    seed_everything,
    set_best,
    signature_settings,
    suppress_output,
    truth_table_from_cm,
)
```

---

## 1. Tensor Validation & Conversion

### Validating Tensors
`is_valid_tensor()` returns `True` for valid tensors or a diagnostic reason string (`"none"`, `"not_tensor"`, `"empty"`, `"nan"`, `"inf"`). **Always compare with `is True`**:

```python
import torch
from positronixlab.utils.utils import is_valid_tensor, assert_tensor

x = torch.randn(4, 8)
if is_valid_tensor(x) is True:
    print("Valid tensor")

# Raises ValueError with diagnostic reason if invalid:
assert_tensor(x, name="latent_activations")
```

### Converting Arbitrary Inputs with `safe_tensor()`
`safe_tensor()` normalizes scalars, tensors, NumPy arrays, and Python sequences into a detached, CPU, at-least-1D `torch.Tensor`:

```python
t1 = safe_tensor(3.14)             # tensor([3.1400])
t2 = safe_tensor([1, 2, 3])        # tensor([1, 2, 3])
t3 = safe_tensor(torch.tensor(7))  # tensor([7])
```

### Inspecting Floating-Point Tensors
Pass `debug=True` explicitly (mutating `utils.debug` does not affect the default argument):

```python
from positronixlab.utils.utils import info_tensor, check_tensor

info_tensor(torch.randn(3, 3), name="weights", debug=True)
check_tensor(torch.randn(3, 3), name="weights", debug=True)
```

> **Note:** `info_tensor()` requires non-empty floating-point (or complex) tensors because it calls `.min()` and `.mean()`.

---

## 2. Reproducibility & Deterministic Hashing

### Seeding RNGs
```python
seed_everything(42)
```
> **Note:** `seed_everything()` currently leaves `torch.backends.cudnn.benchmark = True`. For strict CUDA determinism, follow it with `torch.backends.cudnn.benchmark = False` and `torch.cuda.manual_seed_all(42)`.

### Hashing Config Dictionaries
```python
sig = signature_settings({"lr": 1e-3, "layers": [64, 32]}, algo="sha256")
```
Key ordering does not affect the digest (`sort_keys=True`), and all values must be JSON-serializable.

### Hashing PyTorch Models
```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
digest = model_hash(model)  # SHA-256 hex digest of sorted state_dict keys, shapes, dtypes, and bytes
```

---

## 3. Classification Metrics & Best-Value Tracking

### Confusion Matrix -> Truth Table -> Per-Class F1
Rows are true labels and columns are predicted labels (standard `sklearn` convention):

```python
import numpy as np

cm = np.array([
    [10, 2, 0],
    [1, 15, 1],
    [0, 3, 12],
])
tt = truth_table_from_cm(cm)     # shape (3, 4): columns are (tp, fp, fn, tn)
f1s = f1s_from_truth_table(tt)   # per-class F1 scores
```

### Tracking Best Metrics
```python
bests = {}

# Lower is better (default min=True)
diff, old = set_best(bests, "val_loss", 0.50)       # (1, None)       -> new key
diff, old = set_best(bests, "val_loss", 0.40)       # (0.2, 0.50)     -> 20% relative improvement
diff, old = set_best(bests, "val_loss", 0.45)       # (-1, 0.40)      -> no improvement

# Higher is better (min=False)
diff, old = set_best(bests, "val_acc", 0.80, min=False)  # (1, None)
diff, old = set_best(bests, "val_acc", 0.88, min=False)  # (0.1, 0.80)
```

---

## 4. Multi-Format Data I/O

Supported formats for round-trip `save_data()` / `load_data()` (case-insensitive extensions): `.json`, `.yaml`, `.yml`, `.pkl`, `.pickle`, `.pt`, `.pth`. Parent directories are created automatically, and extra `**kwargs` are forwarded to the underlying serializer/loader.

```python
save_data("artifacts/metrics.json", {"epoch": 5, "loss": 0.123})
metrics = load_data("artifacts/metrics.json")

save_data("artifacts/config.yaml", {"lr": 1e-3, "epochs": 10})
cfg = load_data("artifacts/config.yaml")

save_data("artifacts/weights.pt", {"w": torch.ones(2, 2)})
weights = load_data("artifacts/weights.pt")  # defaults to weights_only=True; override via **kwargs
```

---

## 5. Process & Output Utilities

### Suppressing Native C/C++ Output (e.g. FAISS warnings)
```python
import os

with suppress_output(stderr=True):
    os.write(2, b"noisy native library warning\n")
```
If suppressing Python `print()` on `stdout` (`stderr=False`), flush `sys.stdout` inside the block before exiting so buffered text is written to `/dev/null` rather than leaking after restoration.

### Enforcing Multiprocessing `spawn`
```python
from positronixlab.utils.utils import process_start_spawn

# Call once at entry point before creating CUDA tensors or worker processes:
process_start_spawn()
```

### Session Paths & Naming Helpers
```python
from positronixlab.utils.utils import (
    session_path,
    camel_to_kebab,
    camel_to_snake,
    get_thread_name,
    module_name,
    class_name,
)

# Pass force=True to generate a fresh path instead of returning the cached path
run_dir = session_path("experiment", uid="run01", createDirectory=False, force=True)

camel_to_kebab("ModularSpaceConfig")  # "modular-space-config"
camel_to_snake("ModularSpaceConfig")  # "modular_space_config"
get_thread_name()                     # "mainthread"
class_name(nn.Linear(2, 2))           # "Linear"
module_name(nn.Linear(2, 2))          # "torch.nn.modules.linear"
```
