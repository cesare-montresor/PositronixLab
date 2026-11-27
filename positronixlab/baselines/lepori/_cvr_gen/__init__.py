"""Vendored CVR compositional task renderer.

Attribution:
- Original CVR benchmark: Zerroug et al. (2022), "A Benchmark for Compositional
  Visual Reasoning".
- Compositional subnetwork tasks (103-120): Lepori et al. (NeurIPS 2023),
  "Break It Down: Evidence for Structural Compositionality in Neural Networks"
  (``Compositional_Subnetworks/Vision/CVR/data_generation/``).

Requires ``opencv-python`` (``cv2``) at runtime when generating images on the fly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def is_cvr_renderer_available() -> bool:
    """Return ``True`` if OpenCV (``cv2``) and Pillow (``PIL``) are installed for live CVR rendering."""
    try:
        import cv2  # noqa: F401
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def render_cvr_quartet(
    task_fn_name: str,
    image_size: int = 128,
    seed: int | None = None,
) -> np.ndarray:
    """Render a 4-image CVR Odd-One-Out quartet of shape ``(4, H, W, 3)`` uint8.

    In the returned quartet, index ``3`` is always the odd-one-out image prior
    to permutation by :class:`CVROddOneOutDataset`.
    """
    from positronixlab.baselines.lepori._cvr_gen import subnetwork_tasks
    from positronixlab.baselines.lepori._cvr_gen.utils import render_cv

    fn = getattr(subnetwork_tasks, task_fn_name, None)
    if fn is None:
        raise ValueError(f"Unknown CVR subnetwork task function: {task_fn_name!r}")

    if seed is not None:
        np.random.seed(seed)

    xy, size, shape, color = fn()
    imgs = [
        render_cv(xy[i], size[i], shape[i], color[i], image_size=image_size)
        for i in range(len(shape))
    ]
    return np.stack(imgs, axis=0)
