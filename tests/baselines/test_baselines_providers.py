"""Unit tests for unified SOTA CompositionalDataProviders (Csordas & Lepori)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from positronixlab.baselines import DataProviderLoader
from positronixlab.baselines.csordas import ClassRemovalDataProvider


def test_dataprovider_loader_registry():
    all_ds = DataProviderLoader.list_datasets()
    assert "csordas:tuple_arithmetic" in all_ds
    assert "csordas:add_mul" in all_ds
    assert "csordas:class_removal" in all_ds
    assert "csordas:deepmind_math" in all_ds
    assert "csordas:scan" in all_ds
    assert "csordas:permuted_mnist" in all_ds
    assert "lepori:cvr_inside_contact" in all_ds
    assert "lepori:cvr_inside_count" in all_ds
    assert "lepori:cvr_contact_count" in all_ds
    assert "lepori:cvr_simclr" in all_ds
    assert "lepori:syneval_sv_singular" in all_ds
    assert "lepori:syneval_pro_ant_plural" in all_ds

    with pytest.raises(ValueError, match="Unknown baseline dataset"):
        DataProviderLoader.create("unknown:dataset")


def test_csordas_tuple_arithmetic_and_add_mul_providers():
    ta = DataProviderLoader.create(
        "csordas:tuple_arithmetic", n_train=64, n_val=16, n_test=16, n_digits=1, batch_size=16
    )
    assert ta.list_subtasks() == ["full", "tuple_0", "tuple_1"]
    assert ta.num_features == 40
    assert ta.num_classes == 10
    bx, by = next(iter(ta.train_loader()))
    assert bx.shape == (16, 40)
    assert by.shape == (16,)

    # tuple_0 zeros out tuple_1 channels (second half of 40 features)
    t0_x, _ = next(iter(ta.get_subtask("tuple_0").val_loader()))
    assert (t0_x[:, 20:] == 0).all()
    assert t0_x[:, :20].sum() > 0

    am = DataProviderLoader.create(
        "csordas:add_mul", n_train=64, n_val=16, n_test=16, n_digits=1, batch_size=16
    )
    assert am.list_subtasks() == ["full", "add", "mul"]
    assert am.num_features == 22
    assert am.num_classes == 10
    add_x, _ = next(iter(am.get_subtask("add").val_loader()))
    mul_x, _ = next(iter(am.get_subtask("mul").val_loader()))
    assert (add_x[:, -2] == 1).all() and (add_x[:, -1] == 0).all()
    assert (mul_x[:, -2] == 0).all() and (mul_x[:, -1] == 1).all()


def test_csordas_class_removal_provider():
    x = torch.randn(40, 6)
    y = torch.arange(40) % 4
    ds = TensorDataset(x, y)
    provider = ClassRemovalDataProvider(ds, ds, num_classes=4, batch_size=8)
    assert provider.list_subtasks() == ["all", "without_0", "without_1", "without_2", "without_3"]

    for _, batch_y in provider.get_subtask("without_2").train_loader():
        assert (batch_y != 2).all()


def test_lepori_cvr_and_syneval_compositional_families():
    cvr = DataProviderLoader.create(
        "lepori:cvr_inside_contact",
        n_train=16,
        n_val=8,
        n_test=8,
        image_size=32,
        batch_size=4,
        use_real_generator=False,
    )
    assert set(cvr.list_subtasks()) == {
        "base",
        "inside",
        "contact",
        "both_ooo",
        "ablate_contact",
        "ablate_inside",
    }
    assert set(cvr.eval_matrix_spec().keys()) == {"inside", "contact"}
    bx, by = next(iter(cvr.train_loader()))
    assert bx.shape == (4, 4, 3, 32, 32)
    assert by.shape == (4,)
    assert int(by.min()) >= 0 and int(by.max()) <= 3

    syn = DataProviderLoader.create(
        "lepori:syneval_sv_singular",
        n_train=16,
        n_val=8,
        n_test=8,
        max_length=16,
        batch_size=4,
    )
    assert set(syn.list_subtasks()) == {"base", "s", "v", "ablate_s", "ablate_v"}
    assert syn.get_subtask("ablate_s").train_loader() is None
    sx, sy = next(iter(syn.get_subtask("ablate_s").test_loader()))
    assert sx.shape == (4, 2, 4, 16)
    assert sy.shape == (4,)


def test_paper_calibrated_builders_and_configs():
    from positronixlab.baselines import csordas, lepori
    from positronixlab.dataprovider import DataProviderFactory

    # Csordas paper configs & builders (override sample counts for fast test)
    assert set(csordas.PAPER_CONFIGS.keys()) >= {
        "addmul",
        "tuple_arithmetic",
        "cifar10_class_removal",
        "permuted_mnist",
        "scan",
        "deepmind_math",
    }
    assert csordas.PAPER_CONFIGS["addmul"].dataset_kwargs["n_digits"] == 2
    assert csordas.PAPER_CONFIGS["addmul"].dataset_kwargs["target_mode"] == "digits"

    am = csordas.addmul(n_train=32, n_val=16, n_test=16, batch_size=16)
    assert am.n_digits == 2
    assert am.target_mode == "digits"
    assert am.num_features == 42
    assert am.num_classes == 20
    bx, by = next(iter(am.train))
    assert bx.shape == (16, 42)
    assert by.shape == (16, 2)

    ta = csordas.tuple_arithmetic(n_train=32, n_val=16, n_test=16, batch_size=16)
    assert ta.n_digits == 2
    assert ta.n_tuples == 2
    assert ta.num_features == 80
    tx, ty = next(iter(ta.train))
    assert tx.shape == (16, 80)
    assert ty.shape == (16, 2, 2)

    # Lepori paper configs & builders
    assert set(lepori.PAPER_CONFIGS.keys()) == {
        "cvr_inside_contact",
        "cvr_inside_count",
        "cvr_contact_count",
        "cvr_simclr",
        "syneval_sv_singular",
        "syneval_sv_plural",
        "syneval_pro_ant_singular",
        "syneval_pro_ant_plural",
    }
    assert lepori.PAPER_CONFIGS["cvr_inside_contact"].dataset_kwargs["image_size"] == 128
    assert lepori.PAPER_CONFIGS["cvr_inside_contact"].dataset_kwargs["n_val"] == 500
    assert lepori.PAPER_CONFIGS["cvr_simclr"].dataset_kwargs["n_train_per_task"] == 10_000
    assert lepori.PAPER_CONFIGS["cvr_simclr"].dataset_kwargs["n_val_per_task"] == 500
    assert lepori.PAPER_CONFIGS["syneval_sv_singular"].dataset_kwargs["max_length"] == 50
    assert lepori.PAPER_CONFIGS["syneval_sv_singular"].dataset_kwargs["n_train"] == 9_500
    assert lepori.PAPER_CONFIGS["syneval_sv_singular"].dataset_kwargs["n_val"] == 500
    assert lepori.PAPER_CONFIGS["syneval_pro_ant_singular"].dataset_kwargs["n_train"] == 2_500
    assert lepori.PAPER_CONFIGS["syneval_pro_ant_singular"].dataset_kwargs["n_val"] == 200
    assert lepori.PAPER_CONFIGS["syneval_pro_ant_singular"].dataset_kwargs["n_test"] == 200

    cvr = lepori.cvr_inside_contact(
        n_train=8, n_val=4, n_test=4, batch_size=4, image_size=32, use_real_generator=False
    )
    assert cvr.family == "inside_contact"
    cx, cy = next(iter(cvr.train))
    assert cx.shape == (4, 4, 3, 32, 32)
    assert cy.shape == (4,)

    # DataConfig -> DataProviderFactory round-trip using PAPER_CONFIGS
    cfg = lepori.PAPER_CONFIGS["syneval_sv_plural"].with_updates(
        batch_size=4,
        dataset_kwargs={
            "n_train": 8,
            "n_val": 4,
            "n_test": 4,
            "max_length": 16,
            "download": False,
        },
    )
    syn = DataProviderFactory.from_config(cfg)
    sx, sy = next(iter(syn.train))
    assert sx.shape == (4, 2, 4, 16)
    assert sy.shape == (4,)


def test_cvr_simclr_provider_and_png_strip_loader(tmp_path):
    import numpy as np
    from PIL import Image

    from positronixlab.baselines import lepori
    from positronixlab.dataprovider import (
        CVROddOneOutDataset,
        CVRSimCLRDataProvider,
        DataProviderFactory,
        load_cvr_png_strip,
    )

    # Create a horizontal-strip PNG with 4x 32x32 frames and 2px border padding (im_size=36)
    img_size = 32
    pad = 4
    im_size = img_size + pad
    strip = np.zeros((im_size, 4 * im_size, 3), dtype=np.uint8)
    for k in range(4):
        x0 = k * im_size + pad // 2
        strip[pad // 2 : pad // 2 + img_size, x0 : x0 + img_size, :] = 50 * (k + 1)

    for task_name in (
        "sn_task_1_contact_inside",
        "sn_task_2_inside_count",
        "sn_task_3_contact_count",
    ):
        for split in ("train", "val"):
            split_dir = tmp_path / task_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(4):
                Image.fromarray(strip).save(split_dir / f"{idx:05d}.png")

    # Direct strip loader check
    quartet = load_cvr_png_strip(
        tmp_path / "sn_task_1_contact_inside" / "train" / "00000.png", image_size=img_size
    )
    assert quartet.shape == (4, 3, img_size, img_size)
    assert torch.isclose(quartet[0].mean(), torch.tensor(50.0 / 255.0), atol=1e-3)
    assert torch.isclose(quartet[3].mean(), torch.tensor(200.0 / 255.0), atol=1e-3)

    # CVROddOneOutDataset PNG directory support
    ooo_ds = CVROddOneOutDataset(
        task_fn_name="sn_task_1_contact_inside",
        n_samples=4,
        image_size=img_size,
        png_dir=tmp_path / "sn_task_1_contact_inside" / "train",
        seed=7,
    )
    assert len(ooo_ds) == 4
    sample_x, sample_y = ooo_ds[0]
    assert sample_x.shape == (4, 3, img_size, img_size)
    assert 0 <= int(sample_y) <= 3

    # CVRSimCLRDataProvider from PNG disk directory
    simclr_disk = CVRSimCLRDataProvider(
        n_train_per_task=4,
        n_val_per_task=4,
        image_size=img_size,
        batch_size=4,
        root=tmp_path,
        use_real_generator=False,
    )
    assert len(simclr_disk.dataset_train) == 12
    vx, vy = next(iter(simclr_disk.train))
    assert vx.shape == (4, 2, 2, 3, img_size, img_size)
    assert vy.shape == (4,)
    assert int(vy.min()) >= 0 and int(vy.max()) <= 2

    # CVRSimCLR via DataProviderLoader, lepori.cvr_simclr, and DataProviderFactory (synthetic + as_list)
    simclr_loader = DataProviderLoader.create(
        "lepori:cvr_simclr",
        n_train_per_task=4,
        n_val_per_task=4,
        image_size=img_size,
        batch_size=4,
        use_real_generator=False,
        as_list=True,
    )
    views_list, task_ids = next(iter(simclr_loader.train))
    assert isinstance(views_list, list) and len(views_list) == 2
    assert views_list[0].shape == (4, 2, 3, img_size, img_size)
    assert task_ids.shape == (4,)

    simclr_cfg = lepori.PAPER_CONFIGS["cvr_simclr"].with_updates(
        batch_size=4,
        dataset_kwargs={
            "n_train_per_task": 4,
            "n_val_per_task": 4,
            "image_size": img_size,
            "use_real_generator": False,
        },
    )
    simclr_from_cfg = DataProviderFactory.from_config(simclr_cfg)
    fx, fy = next(iter(simclr_from_cfg.val))
    assert fx.shape == (4, 2, 2, 3, img_size, img_size)
    assert fy.shape == (4,)


def test_syneval_real_generator_from_rawdata_zip(tmp_path):
    import pickle
    import zipfile

    from positronixlab.baselines.lepori._syneval_gen import (
        _ANAPHORA_RAW_FILES,
        _SV_RAW_FILES,
        generate_syneval_family_splits,
        is_syneval_family_cached,
    )
    from positronixlab.dataprovider import SynevalOddOneOutDataProvider

    # Build a miniature RawData.zip matching Marvin & Linzen pickle schema
    raw_zip = tmp_path / "RawData.zip"
    with zipfile.ZipFile(raw_zip, "w") as zf:
        for fname in _SV_RAW_FILES:
            for prefix in ("singular_", "plural_"):
                pairs = [
                    (f"{prefix}{fname}_gram_{i}", f"{prefix}{fname}_ungram_{i}")
                    for i in range(60)
                ]
                zf.writestr(f"RawData/{prefix}{fname}", pickle.dumps({"cat": pairs}))
        for fname in _ANAPHORA_RAW_FILES:
            for prefix in ("singular_", "plural_"):
                pairs = [
                    (f"{prefix}{fname}_gram_{i}", f"{prefix}{fname}_ungram_{i}")
                    for i in range(120)
                ]
                zf.writestr(f"RawData/{prefix}{fname}", pickle.dumps({"cat": pairs}))

    out_dir = tmp_path / "syneval_cache"
    generate_syneval_family_splits(
        "sv_singular",
        raw_zip,
        out_dir,
        seed=0,
        train_size=12,
        val_size=4,
        test_size=4,
        part_val_size=4,
        part_test_size=4,
    )
    assert is_syneval_family_cached("sv_singular", out_dir)

    # Dummy deterministic tokenizer so test runs offline without HuggingFace downloads
    def dummy_tokenizer(sents, max_length=16, padding="max_length", truncation=True, return_tensors="pt"):
        n = len(sents)
        ids = torch.zeros((n, max_length), dtype=torch.long)
        attn = torch.zeros((n, max_length), dtype=torch.long)
        for i, s in enumerate(sents):
            toks = [1 + (ord(c) % 50) for c in s[:max_length]]
            ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
            attn[i, : len(toks)] = 1
        return {"input_ids": ids, "attention_mask": attn}

    provider = SynevalOddOneOutDataProvider(
        family="sv_singular",
        n_train=12,
        n_val=4,
        n_test=4,
        max_length=16,
        batch_size=4,
        root=out_dir,
        tokenizer=dummy_tokenizer,
        download=False,
    )
    bx, by = next(iter(provider.train))
    assert bx.shape == (4, 2, 4, 16)
    assert by.shape == (4,)
    assert int(by.min()) >= 0 and int(by.max()) <= 3
    assert provider.get_subtask("ablate_s").train_loader() is None
    ax, ay = next(iter(provider.get_subtask("ablate_s").test_loader()))
    assert ax.shape == (4, 2, 4, 16)
    assert ay.shape == (4,)

    # Also test Reflexive Anaphora generation
    generate_syneval_family_splits(
        "pro_ant_plural",
        raw_zip,
        out_dir,
        seed=1,
        train_size=12,
        val_size=4,
        test_size=4,
        part_val_size=4,
        part_test_size=4,
    )
    assert is_syneval_family_cached("pro_ant_plural", out_dir)
    anaphora_provider = SynevalOddOneOutDataProvider(
        family="pro_ant_plural",
        n_train=12,
        n_val=4,
        n_test=4,
        max_length=16,
        batch_size=4,
        root=out_dir,
        tokenizer=dummy_tokenizer,
        download=False,
    )
    px, py = next(iter(anaphora_provider.train))
    assert px.shape == (4, 2, 4, 16)
    assert py.shape == (4,)
