"""Unit tests for unified SOTA CompositionalDataProviders (Csordas & Lepori)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from positronixlab.dataprovider.sota import (
    ClassRemovalDataProvider,
    DataProviderLoader,
)


def test_dataprovider_loader_registry():
    all_ds = DataProviderLoader.list_datasets()
    assert "csordas:tuple_arithmetic" in all_ds
    assert "csordas:add_mul" in all_ds
    assert "csordas:class_removal" in all_ds
    assert "lepori:cvr_inside_contact" in all_ds
    assert "lepori:cvr_inside_count" in all_ds
    assert "lepori:cvr_contact_count" in all_ds
    assert "lepori:syneval_sv_singular" in all_ds
    assert "lepori:syneval_pro_ant_plural" in all_ds

    with pytest.raises(ValueError, match="Unknown SOTA dataset"):
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
