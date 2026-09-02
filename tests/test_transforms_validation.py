import numpy as np
import pytest
import torch

from echo_research.data.transforms import AugmentState, EchoTransform


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizontal_flip_prob": 1.1},
        {"rotation_deg": -1},
        {"translation_fraction": 1.0},
        {"scale_jitter": 1.0},
        {"gamma_jitter": 1.0},
        {"contrast": -0.1},
    ],
)
def test_invalid_augmentation_ranges_fail_fast(kwargs):
    with pytest.raises(ValueError):
        EchoTransform(**kwargs)


def test_high_dice_augmentation_ranges_are_valid():
    transform = EchoTransform(
        horizontal_flip_prob=0.5,
        rotation_deg=8,
        translation_fraction=0.03,
        scale_jitter=0.08,
        brightness=0.08,
        contrast=0.10,
        gamma_jitter=0.12,
    )
    state = transform.sample_state()
    assert 0.92 <= state.scale <= 1.08
    assert 0.88 <= state.gamma <= 1.12


def test_reused_state_keeps_phase_geometry_synchronized():
    transform = EchoTransform(image_size=(32, 32), augment=True, rotation_deg=8, translation_fraction=0.1)
    image = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 10:22] = 1.0
    state = transform.sample_state()
    image_a, mask_a = transform(image, mask, task="binary", state=state)
    image_b, mask_b = transform(image, mask, task="binary", state=state)
    assert torch.equal(image_a, image_b)
    assert torch.equal(mask_a, mask_b)


def test_photometric_adjustment_does_not_brighten_affine_fill():
    transform = EchoTransform(image_size=(32, 32), augment=True)
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    mask = np.ones((32, 32), dtype=np.float32)
    state = AugmentState(False, 0.0, 8, 0, 1.0, 0.2, 1.0, 1.0)
    transformed, _ = transform(image, mask, task="binary", state=state)
    assert torch.count_nonzero(transformed[:, :, :8]) == 0
