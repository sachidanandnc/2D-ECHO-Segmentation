from pathlib import Path

from echo_research.config import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_region_only_control_is_segmentation_only():
    cfg = load_config(CONFIG_DIR / "high_dice_region_only.yaml")
    assert cfg["loss"]["region_weight"] == 1.0
    assert cfg["loss"]["dice_weight"] == 2.0
    assert cfg["loss"]["adversarial_weight"] == 0.0
    assert cfg["loss"]["pixel_l1_weight"] == 0.0
    assert cfg["loss"]["boundary_weight"] == 0.0
    assert cfg["loss"]["functional_weight"] == 0.0
    assert cfg["model"]["normalization"] == "instance"


def test_high_dice_comparison_keeps_gan_conservative():
    cfg = load_config(CONFIG_DIR / "high_dice_binary.yaml")
    assert cfg["loss"]["dice_weight"] > cfg["loss"]["adversarial_weight"]
    assert cfg["training"]["adversarial_warmup_epochs"] == 15
    assert cfg["training"]["d_update_every"] == 2
    assert cfg["training"]["lr_d"] < cfg["training"]["lr_g"]
    assert cfg["evaluation"]["original_grid"] is True
    assert cfg["evaluation"]["require_calibration_policy"] is True
