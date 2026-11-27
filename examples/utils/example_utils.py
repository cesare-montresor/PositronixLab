"""Runnable CPU example demonstrating positronixlab.utils.utils helpers."""

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from positronixlab.utils.utils import (
    assert_tensor,
    camel_to_kebab,
    camel_to_snake,
    class_name,
    f1s_from_truth_table,
    get_thread_name,
    info_tensor,
    is_valid_tensor,
    load_data,
    model_hash,
    module_name,
    safe_tensor,
    save_data,
    seed_everything,
    set_best,
    signature_settings,
    suppress_output,
    truth_table_from_cm,
)


def main():
    # 1. Reproducible seeding
    seed_everything(42)
    print(f"[thread] {get_thread_name()}")

    # 2. Naming helpers
    name = "ModularSpaceConfig"
    print(f"[naming] {name} -> kebab={camel_to_kebab(name)}, snake={camel_to_snake(name)}")

    # 3. Tensor validation, inspection, and safe conversion
    x = torch.randn(3, 4)
    assert_tensor(x, name="activations")
    print(f"[valid] x={is_valid_tensor(x)}, nan={is_valid_tensor(torch.tensor([float('nan')]))}")
    info_tensor(x, name="activations", debug=True)
    print(f"[safe_tensor] scalar -> {safe_tensor(3.14)}, list -> {safe_tensor([1, 2, 3])}")

    # 4. Confusion matrix -> truth table -> per-class F1
    cm = np.array([[10, 2, 0], [1, 15, 1], [0, 3, 12]])
    tt = truth_table_from_cm(cm)
    f1s = f1s_from_truth_table(tt)
    print(f"[metrics] per-class F1: {np.round(f1s, 4).tolist()}")

    # 5. Best-metric tracking
    bests = {}
    for step, loss in enumerate([0.90, 0.72, 0.75, 0.54], start=1):
        rel_diff, prev = set_best(bests, "val_loss", loss, min=True)
        print(f"[best] step={step} loss={loss:.2f} -> rel_diff={rel_diff:.3f}, prev={prev}")

    # 6. Deterministic settings & model hashing
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    print(f"[inspect] {module_name(model)}.{class_name(model)}")
    print(f"[hash] settings={signature_settings({'lr': 1e-3, 'epochs': 10})}")
    print(f"[hash] model={model_hash(model)[:16]}...")

    # 7. Multi-format I/O in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "summary.json")
        pt_path = os.path.join(tmpdir, "weights.pt")
        save_data(json_path, {"best_loss": bests["val_loss"], "f1s": f1s.tolist()})
        save_data(pt_path, {"sample": torch.tensor([1.0, 2.0])})
        print(f"[io] loaded json: {load_data(json_path)}")
        print(f"[io] loaded pt:   {load_data(pt_path)}")

    # 8. C-level stderr suppression
    with suppress_output(stderr=True):
        os.write(2, b"This native stderr message is suppressed.\n")
    print("[suppress_output] completed cleanly.")


if __name__ == "__main__":
    main()
