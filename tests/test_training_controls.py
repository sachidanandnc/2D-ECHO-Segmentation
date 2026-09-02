import pytest
import torch
from torch import nn

from echo_research.engine.trainer import Trainer
from echo_research.factory import build_model


def _config(adversarial_weight=0.05):
    return {
        "experiment": {"seed": 1},
        "data": {"task": "binary", "paired_phases": False},
        "model": {"base_channels": 8, "num_classes": 4, "normalization": "instance"},
        "loss": {
            "region_weight": 1.0,
            "ce_weight": 1.0,
            "dice_weight": 1.0,
            "adversarial_weight": adversarial_weight,
        },
        "training": {
            "lr_g": 1e-4,
            "lr_d": 2.5e-5,
            "adversarial_warmup_epochs": 2,
            "adversarial_ramp_epochs": 4,
        },
        "evaluation": {"threshold": 0.5},
    }


def test_adversarial_warmup_and_ramp(tmp_path):
    cfg = _config(0.08)
    trainer = Trainer(build_model(cfg), cfg, [], [], torch.device("cpu"), tmp_path)
    assert trainer._adversarial_weight(1) == 0.0
    assert trainer._adversarial_weight(2) == 0.0
    assert trainer._adversarial_weight(3) == pytest.approx(0.02)
    assert trainer._adversarial_weight(6) == pytest.approx(0.08)
    assert trainer._adversarial_weight(20) == pytest.approx(0.08)


def test_zero_adversarial_weight_stays_off(tmp_path):
    cfg = _config(0.0)
    trainer = Trainer(build_model(cfg), cfg, [], [], torch.device("cpu"), tmp_path)
    assert trainer._adversarial_weight(100) == 0.0


def test_instance_norm_is_selected_for_batch_size_one():
    cfg = _config()
    model = build_model(cfg)
    modules = list(model.modules())
    assert any(isinstance(module, nn.InstanceNorm2d) for module in modules)
    assert not any(isinstance(module, nn.BatchNorm2d) for module in modules)
    output = model.generator(torch.randn(1, 1, 256, 256))
    assert output.shape == (1, 1, 256, 256)
    assert torch.isfinite(output).all()


def test_unknown_normalization_fails_clearly():
    cfg = _config()
    cfg["model"]["normalization"] = "mystery"
    with pytest.raises(ValueError, match="Unknown normalization"):
        build_model(cfg)
