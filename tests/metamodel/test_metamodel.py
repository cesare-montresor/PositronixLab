from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from positronixlab.metamodel import (
    BaseModel,
    MetaCriterion,
    MetaMetric,
    MetaModel,
    MetaModelConfig,
    MetaTransform,
    ResNet,
)

# ---------------------------------------------------------------------------
# MetaModelConfig
# ---------------------------------------------------------------------------

class TestMetaModelConfig:
    def test_defaults_and_signature(self):
        cfg1 = MetaModelConfig()
        cfg2 = MetaModelConfig()
        assert cfg1.task == "classification"
        assert cfg1.criterion_type == "cross_entropy"
        assert cfg1.model_type == "fc"
        assert cfg1.signature() == cfg2.signature()

        cfg3 = cfg1.with_updates(input_hidden_dim=64)
        assert cfg3.input_hidden_dim == 64
        assert cfg1.input_hidden_dim == 32
        assert cfg3.signature() != cfg1.signature()

    def test_from_dict_coercion_and_unknown_key_drop(self):
        cfg = MetaModelConfig.from_dict({
            "task": "regression",
            "input_data_shape": "128",
            "input_batch_norm": "false",
            "dropout": "0.25",
            "unknown_extra_key": 999,
        })
        assert cfg.task == "regression"
        assert cfg.input_data_shape == 128
        assert cfg.input_batch_norm is False
        assert cfg.dropout == pytest.approx(0.25)
        assert not hasattr(cfg, "unknown_extra_key")

    def test_save_and_load_yaml_and_json(self, tmp_path):
        cfg = MetaModelConfig(name="exp1", input_data_shape=64, output_data_shape=5)
        for ext in ("yaml", "json"):
            path = str(tmp_path / f"cfg.{ext}")
            cfg.save(path)
            loaded = MetaModelConfig.load(path)
            assert loaded == cfg


# ---------------------------------------------------------------------------
# BaseModel & ResNet
# ---------------------------------------------------------------------------

class _TinyModel(BaseModel):
    def __init__(self, name=None, device=None):
        super().__init__(name=name, device=device)
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class TestBaseModel:
    def test_init_defaults_and_custom_name(self):
        m = _TinyModel(device=torch.device("cpu"))
        assert m.name == "_TinyModel"
        assert m.device == torch.device("cpu")
        assert m.basepath is None

        m2 = _TinyModel(name="custom_net", device=torch.device("cpu"))
        assert m2.name == "custom_net"

    def test_model_path_and_save_load_roundtrip(self, tmp_path):
        m = _TinyModel(name="tiny", device=torch.device("cpu"))
        m.basepath = str(tmp_path)
        expected_path = str((tmp_path / "tiny.pth").resolve())
        assert m.model_path() == expected_path

        saved = m.save()
        assert saved == expected_path

        m2 = _TinyModel(name="tiny", device=torch.device("cpu"))
        with torch.no_grad():
            m2.linear.weight.zero_()
            m2.linear.bias.zero_()

        loaded = m2.load(saved)
        assert loaded == saved
        assert torch.allclose(m.linear.weight, m2.linear.weight)
        assert torch.allclose(m.linear.bias, m2.linear.bias)

    def test_load_missing_file_returns_none(self, tmp_path):
        m = _TinyModel(name="missing", device=torch.device("cpu"))
        result = m.load(str(tmp_path / "does_not_exist.pth"))
        assert result is None


class TestResNet:
    def test_residual_skip_with_module_and_ordered_dict(self):
        lin = nn.Linear(4, 4, bias=False)
        nn.init.eye_(lin.weight)
        res = ResNet(lin)
        x = torch.ones(2, 4)
        assert torch.allclose(res(x), 2.0 * x)

        od = OrderedDict([("linear", lin)])
        res_od = ResNet(od)
        assert isinstance(res_od.module, nn.Sequential)
        assert torch.allclose(res_od(x), 2.0 * x)


# ---------------------------------------------------------------------------
# MetaCriterion & MetaMetric
# ---------------------------------------------------------------------------

class TestMetaCriterion:
    @pytest.mark.parametrize(
        "criterion_type,y_hat,y",
        [
            ("cross_entropy", torch.randn(4, 3), torch.tensor([0, 1, 2, 1])),
            ("mse", torch.randn(4, 2), torch.randn(4, 2)),
            ("mae", torch.randn(4, 2), torch.randn(4, 2)),
            ("huber", torch.randn(4, 2), torch.randn(4, 2)),
            ("bce_with_logits", torch.randn(4, 1), torch.randint(0, 2, (4, 1)).float()),
            ("bce", torch.rand(4, 1), torch.randint(0, 2, (4, 1)).float()),
        ],
    )
    def test_builtin_losses(self, criterion_type, y_hat, y):
        cfg = MetaModelConfig(task="classification", criterion_type=criterion_type)
        crit = MetaCriterion(cfg)
        val = crit.loss(None, y, y_hat)
        assert val.ndim == 0
        assert torch.isfinite(val)

    def test_autoencoder_overrides_target_with_input(self):
        cfg = MetaModelConfig(task="autoencoder", criterion_type="mse")
        crit = MetaCriterion(cfg)
        x = torch.ones(4, 6)
        y_hat = torch.ones(4, 6) * 3.0
        # y is intentionally different; autoencoder must compare y_hat against x
        val = crit.loss(x, torch.zeros(4, 6), y_hat)
        assert val.item() == pytest.approx(4.0)

    def test_custom_loss_and_unsupported_error(self):
        cfg = MetaModelConfig(criterion_type="my_l1")
        crit = MetaCriterion(cfg)
        crit.custom_loss("my_l1", lambda x, y, y_hat: (y_hat - y).abs().mean())
        val = crit.loss(None, torch.zeros(3), torch.full((3,), 2.0))
        assert val.item() == pytest.approx(2.0)

        bad = MetaCriterion(MetaModelConfig(criterion_type="nonexistent"))
        with pytest.raises(ValueError, match="Unsupported criterion type"):
            bad.loss(None, torch.zeros(2), torch.zeros(2))


class TestMetaMetric:
    def test_default_metric_selection_and_computation(self):
        cls_metric = MetaMetric(MetaModelConfig(task="classification"))
        assert cls_metric.metric_type == "accuracy"
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 0.0]])
        targets = torch.tensor([0, 1, 1, 0])
        acc = cls_metric.metric(None, targets, logits)
        assert acc.item() == pytest.approx(0.75)
        assert not acc.requires_grad

        seg_metric = MetaMetric(MetaModelConfig(task="segmentation"))
        assert seg_metric.metric_type == "ioU"
        pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
        true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        iou = seg_metric.metric(None, true, pred)
        assert iou.item() == pytest.approx(1.0 / 3.0, abs=1e-5)

        reg_metric = MetaMetric(MetaModelConfig(task="regression"))
        assert reg_metric.metric_type == "f1_score"
        f1 = reg_metric.metric(None, true, pred)
        assert f1.item() == pytest.approx(0.5, abs=1e-5)

    def test_unsupported_metric_raises(self):
        m = MetaMetric(MetaModelConfig(), metric_type="unknown_metric")
        with pytest.raises(ValueError, match="Unsupported metric type"):
            m.metric(None, torch.zeros(2), torch.zeros(2))

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metacriterion.py:45,57: self.custom_metric = {} shadows def custom_metric()",
    )
    def test_custom_metric_registration_bug(self):
        m = MetaMetric(MetaModelConfig(), metric_type="my_metric")
        m.custom_metric("my_metric", lambda x, y, y_hat: torch.tensor(1.0))
        assert m.metric(None, None, None).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetaModel
# ---------------------------------------------------------------------------

class TestMetaModel:
    def test_fc_classification_forward_loss_and_utilities(self):
        cfg = MetaModelConfig(
            task="classification",
            criterion_type="cross_entropy",
            model_type="fc",
            input_data_shape=16,
            input_hidden_dim=32,
            input_num_layers=3,
            output_data_shape=5,
            output_num_layers=2,
            input_dropout_every=-1,
            output_dropout_every=-1,
            input_residual=True,
            output_residual=True,
        )
        model = MetaModel("cls_fc", cfg)
        model.to_cpu()

        assert len(model.nn_input) == 3
        assert isinstance(model.nn_latent, nn.Identity)
        assert len(model.nn_output) == 2
        assert isinstance(model.nn_input[1], ResNet)
        assert model.last_layer_size() == (32, 5)
        assert model.get_model_size() > 0
        assert model.nmum_fop() > 0

        x = torch.randn(4, 16)
        y = torch.randint(0, 5, (4,))
        y_hat = model(x)
        assert y_hat.shape == (4, 5)

        loss = model.loss(x, y, y_hat)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_fc_autoencoder_mirrors_encoder_and_reconstructs(self):
        cfg = MetaModelConfig(
            task="autoencoder",
            criterion_type="mse",
            model_type="fc",
            input_data_shape=24,
            input_hidden_dim=16,
            input_num_layers=3,
            input_reduce_every=2,
            input_dropout_every=-1,
            output_dropout_every=-1,
        )
        model = MetaModel("ae_fc", cfg)
        model.to_cpu()

        assert len(model.nn_output) == len(model.nn_input) == 3
        x = torch.randn(4, 24)
        recon = model(x)
        assert recon.shape == x.shape
        loss = model.loss(x, None, recon)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh", "leaky_relu"])
    def test_supported_activations(self, activation):
        cfg = MetaModelConfig(
            input_data_shape=8,
            input_hidden_dim=8,
            input_num_layers=1,
            output_data_shape=4,
            output_num_layers=2,
            activation=activation,
            input_dropout_every=-1,
            output_dropout_every=-1,
        )
        model = MetaModel("act_test", cfg)
        model.to_cpu()
        out = model(torch.randn(3, 8))
        assert out.shape == (3, 4)

    def test_invalid_task_model_type_activation_raise(self):
        with pytest.raises(ValueError, match="Task must be one of"):
            MetaModel("bad", MetaModelConfig(task="segmentation"))

        with pytest.raises(ValueError, match="Model type must be one of"):
            MetaModel("bad", MetaModelConfig(model_type="gnn"))

        with pytest.raises(ValueError, match="Unsupported activation"):
            MetaModel("bad", MetaModelConfig(activation="gelu"))

    # --- Confirmed bug xfail tests ---

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:61-72: MetaModel.__init__ ignores device arg and unconditionally calls to_gpu()",
    )
    def test_init_respects_cpu_device_arg(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable; bug only manifests when CUDA is available")
        cfg = MetaModelConfig(input_data_shape=8, input_hidden_dim=8, output_data_shape=4)
        model = MetaModel("cpu_req", cfg, device=torch.device("cpu"))
        assert model.device == torch.device("cpu")

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:250-255: output_num_layers=1 hits is_first_layer before is_last_layer, outputting input_hidden_dim",
    )
    def test_single_output_layer_outputs_output_data_shape(self):
        cfg = MetaModelConfig(
            input_data_shape=16,
            input_hidden_dim=32,
            input_num_layers=2,
            output_data_shape=5,
            output_num_layers=1,
            input_dropout_every=-1,
            output_dropout_every=-1,
        )
        model = MetaModel("single_out", cfg)
        model.to_cpu()
        y_hat = model(torch.randn(4, 16))
        assert y_hat.shape == (4, 5)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:252: build_model_output_block uses input_hidden_dim instead of output_hidden_dim",
    )
    def test_output_hidden_dim_is_used(self):
        cfg = MetaModelConfig(
            input_data_shape=16,
            input_hidden_dim=32,
            input_num_layers=2,
            output_hidden_dim=12,
            output_data_shape=5,
            output_num_layers=2,
            input_dropout_every=-1,
            output_dropout_every=-1,
        )
        model = MetaModel("out_hid", cfg)
        model.to_cpu()
        first_out_block = model.nn_output[0]
        if isinstance(first_out_block, ResNet):
            first_out_block = first_out_block.module
        assert first_out_block.layer.out_features == 12

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:264-276: CNN classification attaches BatchNorm2d after Linear and lacks Flatten",
    )
    def test_cnn_classification_forward(self):
        cfg = MetaModelConfig(
            task="classification",
            model_type="cnn",
            input_data_shape=3,
            input_hidden_dim=8,
            input_num_layers=2,
            output_data_shape=4,
            output_num_layers=2,
            input_dropout_every=-1,
            output_dropout_every=-1,
        )
        model = MetaModel("cnn_cls", cfg)
        model.to_cpu()
        y_hat = model(torch.randn(2, 3, 8, 8))
        assert y_hat.shape == (2, 4)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:302,390-403: CNN autoencoder with >=2 layers calls layer_size() on ConvTranspose2d and crashes",
    )
    def test_cnn_autoencoder_multilayer_builds(self):
        cfg = MetaModelConfig(
            task="autoencoder",
            criterion_type="mse",
            model_type="cnn",
            input_data_shape=3,
            input_hidden_dim=8,
            input_num_layers=2,
            input_dropout_every=-1,
            output_dropout_every=-1,
        )
        model = MetaModel("cnn_ae", cfg)
        model.to_cpu()
        x = torch.randn(2, 3, 8, 8)
        assert model(x).shape == x.shape

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:222,279: dropout_every=0 enables dropout on layer 0 instead of disabling dropout",
    )
    def test_dropout_every_zero_disables_dropout(self):
        cfg = MetaModelConfig(
            input_data_shape=8,
            input_hidden_dim=8,
            input_num_layers=2,
            output_data_shape=4,
            output_num_layers=2,
            input_dropout_every=0,
            output_dropout_every=0,
        )
        model = MetaModel("no_drop", cfg)
        assert not any(isinstance(m, nn.Dropout) for m in model.modules())

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:420: nmum_fop() indexes kernel_size[1] on nn.Conv1d causing IndexError",
    )
    def test_nmum_fop_handles_conv1d(self):
        cfg = MetaModelConfig(input_data_shape=8, input_hidden_dim=8, output_data_shape=4)
        model = MetaModel("conv1d_flops", cfg)
        model.nn_latent = nn.Conv1d(8, 8, kernel_size=3)
        assert model.nmum_fop() > 0

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:392-403: layer_size(resnet_block) unwraps to nn.Sequential instead of inner layer",
    )
    def test_layer_size_accepts_resnet_block(self):
        cfg = MetaModelConfig(
            input_data_shape=8,
            input_hidden_dim=8,
            input_num_layers=2,
            output_data_shape=4,
            input_residual=True,
        )
        model = MetaModel("res_size", cfg)
        resnet_block = model.nn_input[1]
        assert isinstance(resnet_block, ResNet)
        assert model.layer_size(resnet_block) == (8, 8)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:164: hardcoded padding=1 shrinks spatial dims and breaks ResNet skip when cnn_kernel_size != 3",
    )
    def test_cnn_kernel_size_5_with_residual(self):
        cfg = MetaModelConfig(
            task="classification",
            model_type="cnn",
            input_data_shape=3,
            input_hidden_dim=8,
            input_num_layers=2,
            output_data_shape=4,
            output_num_layers=2,
            cnn_kernel_size=5,
            input_residual=True,
            input_dropout_every=-1,
            output_dropout_every=-1,
        )
        model = MetaModel("cnn_k5", cfg)
        model.to_cpu()
        x = torch.randn(2, 3, 16, 16)
        out = model.nn_input(x)
        assert out.shape == (2, 8, 16, 16)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug in metamodel.py:58,158-172: 'rnn' is listed in LAYER_TYPES but unimplemented in block_layer()",
    )
    def test_rnn_listed_in_layer_types_builds(self):
        cfg = MetaModelConfig(model_type="rnn", input_data_shape=8, input_hidden_dim=8)
        MetaModel("rnn_net", cfg)


# ---------------------------------------------------------------------------
# MetaTransform
# ---------------------------------------------------------------------------

class TestMetaTransform:
    def test_pipeline_normalize_flatten_dtype(self):
        t = MetaTransform(["normalize", "flatten", "to_float"])
        img = torch.ones(1, 4, 4)
        out = t.apply(img)
        assert out.shape == (16,)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.ones(16))

    def test_to_long_and_normalize_image_and_augment_image(self):
        t_long = MetaTransform(["to_long"])
        assert t_long.apply(torch.tensor([1.0, 2.0])).dtype == torch.int64

        t_img = MetaTransform(["normalize_image"])
        rgb = torch.rand(3, 8, 8)
        assert t_img.apply(rgb).shape == (3, 8, 8)

        t_aug = MetaTransform(["augment_image"])
        assert t_aug.apply(rgb).shape == (3, 8, 8)

    def test_unsupported_transform_raises(self):
        with pytest.raises(ValueError, match="Unsupported transform type"):
            MetaTransform(["does_not_exist"])

    def test_metatransform_is_callable(self):
        t = MetaTransform(["flatten"])
        out = t(torch.ones(2, 3))
        assert out.shape == (6,)
