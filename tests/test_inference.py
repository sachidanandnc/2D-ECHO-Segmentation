import hashlib
import json

import numpy as np
import pytest
import torch

from echo_research.inference import (
    binary_inference_options,
    binary_masks_from_logits,
    binary_masks_from_probabilities,
    postprocess_binary_mask,
)
from echo_research.run_manifest import config_sha256, patient_ids_sha256


def test_threshold_boundary_and_range_validation():
    probabilities = np.array([[[0.20, 0.49, 0.50, 0.80]]], dtype=np.float32)
    mask = binary_masks_from_probabilities(probabilities, threshold=0.50)
    assert mask.tolist() == [[[False, False, True, True]]]
    with pytest.raises(ValueError):
        binary_masks_from_probabilities(probabilities, threshold=1.01)
    with pytest.raises(ValueError, match="NaN"):
        binary_masks_from_probabilities(np.array([[[np.nan]]], dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_masks_from_probabilities(np.array([[[1.1]]], dtype=np.float32))


def test_logits_are_sigmoided_once():
    logits = torch.tensor([[[[0.4]]]])
    assert binary_masks_from_logits(logits, threshold=0.55).item()
    assert not binary_masks_from_logits(logits, threshold=0.65).item()


def test_largest_component_and_hole_fill():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:6, 1:6] = True
    mask[3, 3] = False
    mask[7, 7] = True
    result = postprocess_binary_mask(mask, keep_largest_component=True, fill_holes=True)
    assert result.sum() == 25
    assert result[3, 3]
    assert not result[7, 7]


def test_largest_component_uses_eight_connectivity():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[5, 5] = True
    result = postprocess_binary_mask(mask, keep_largest_component=True)
    assert result.sum() == 2
    assert result[0, 0] and result[1, 1]


def test_postprocessing_disabled_preserves_mask():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[3, 3] = True
    np.testing.assert_array_equal(postprocess_binary_mask(mask), mask)


def test_calibration_policy_is_bound_to_checkpoint_and_split(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    manifest = {"train": ["p1"], "val": ["p2", "p3"], "test": ["p4"]}
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    policy = {
        "schema_version": 1,
        "selection_split": "validation",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_sha256": config_sha256({}),
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "validation_patient_ids_sha256": patient_ids_sha256(manifest["val"]),
        "best_threshold": 0.47,
        "keep_largest_component": True,
        "fill_holes": True,
    }
    policy_path = tmp_path / "threshold_calibration.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    assert binary_inference_options({}, policy_path, checkpoint) == (0.47, True, True)
    manifest_path.write_text(json.dumps({**manifest, "val": ["p9"]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="split-manifest hash"):
        binary_inference_options({}, policy_path, checkpoint)


def test_inference_config_hash_ignores_paths_and_batch_mechanics():
    first = {
        "experiment": {"name": "run", "seed": 7, "config_path": "/old/config.yaml"},
        "data": {"root": "/old/data", "task": "binary"},
        "model": {"base_channels": 64},
        "loss": {"dice_weight": 2.0},
        "training": {"batch_size": 2},
        "evaluation": {"threshold": 0.5},
    }
    relocated = {
        **first,
        "experiment": {**first["experiment"], "config_path": "/new/config.yaml"},
        "data": {**first["data"], "root": "/new/data"},
        "training": {"batch_size": 1},
    }
    assert config_sha256(first) == config_sha256(relocated)
    changed_policy = {**relocated, "evaluation": {"threshold": 0.6}}
    assert config_sha256(first) != config_sha256(changed_policy)
