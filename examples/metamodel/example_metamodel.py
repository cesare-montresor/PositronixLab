"""
Concise runnable CPU example for `positronixlab.metamodel`.

Demonstrates:
1. Configuring `MetaModelConfig` (immutable updates + deterministic signature)
2. Training a small FC classification `MetaModel` on synthetic CPU data
3. Running an FC autoencoder `MetaModel`
4. Registering a custom loss on `MetaCriterion`
5. Transforming tensors with `MetaTransform`
6. Saving and reloading checkpoints via `BaseModel.save()` / `load()`
"""

import tempfile
import torch

from positronixlab.metamodel import (
    MetaCriterion,
    MetaMetric,
    MetaModel,
    MetaModelConfig,
    MetaTransform,
)


def main() -> None:
    torch.manual_seed(0)

    # 1. Classification MetaModel on CPU
    cls_cfg = MetaModelConfig(
        name="demo_classifier",
        task="classification",
        criterion_type="cross_entropy",
        model_type="fc",
        input_data_shape=16,
        input_hidden_dim=32,
        input_num_layers=2,
        output_data_shape=3,
        output_num_layers=2,
        input_dropout_every=-1,   # disable dropout
        output_dropout_every=-1,
    )
    print(f"[Config] {cls_cfg.name} signature={cls_cfg.signature()[:10]}...")

    model = MetaModel(cls_cfg.name, cls_cfg)
    model.to_cpu()
    metric = MetaMetric(cls_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(32, 16)
    y = torch.randint(0, 3, (32,))

    for step in range(5):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = model.loss(x, y, y_hat)
        loss.backward()
        optimizer.step()
    acc = metric.metric(x, y, model(x))
    print(f"[Classification] final_loss={loss.item():.4f}, acc={acc.item():.2%}, size={model.get_model_size():.4f} MB")

    # 2. Autoencoder MetaModel on CPU
    ae_cfg = cls_cfg.with_updates(
        name="demo_autoencoder",
        task="autoencoder",
        criterion_type="mse",
        input_hidden_dim=8,
    )
    ae = MetaModel(ae_cfg.name, ae_cfg)
    ae.to_cpu()
    recon = ae(x)
    ae_loss = ae.loss(x, None, recon)
    print(f"[Autoencoder] input={tuple(x.shape)} -> recon={tuple(recon.shape)}, mse={ae_loss.item():.4f}")

    # 3. Custom loss via MetaCriterion
    custom_cfg = MetaModelConfig(task="regression", criterion_type="scaled_l1")
    criterion = MetaCriterion(custom_cfg)
    criterion.custom_loss("scaled_l1", lambda _x, target, pred: 2.0 * (pred - target).abs().mean())
    custom_val = criterion.loss(None, torch.zeros(4, 2), torch.ones(4, 2))
    print(f"[MetaCriterion] custom scaled_l1={custom_val.item():.4f}")

    # 4. MetaTransform pipeline
    transform = MetaTransform(["normalize", "flatten", "to_float"])
    sample_img = torch.rand(1, 4, 4)
    flat_vec = transform.apply(sample_img)
    print(f"[MetaTransform] {tuple(sample_img.shape)} -> {tuple(flat_vec.shape)} ({flat_vec.dtype})")

    # 5. Checkpoint save / load round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        model.basepath = tmpdir
        ckpt_path = model.save()
        reloaded = MetaModel(cls_cfg.name, cls_cfg)
        reloaded.to_cpu()
        reloaded.load(ckpt_path)
        model.eval()
        reloaded.eval()
        assert torch.allclose(model(x), reloaded(x))
        print(f"[Checkpoint] saved & verified reload from {ckpt_path}")


if __name__ == "__main__":
    main()
