from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, label

from .run_manifest import config_sha256, patient_ids_sha256, run_split_manifest_path


def validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Binary threshold must be in [0, 1], got {threshold}")
    return threshold


def postprocess_binary_mask(
    mask: np.ndarray,
    keep_largest_component: bool = False,
    fill_holes: bool = False,
) -> np.ndarray:
    """Apply deterministic LV-cavity postprocessing to one 2-D binary mask."""
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError(f"Expected a 2-D binary mask, got shape {result.shape}")
    if keep_largest_component and result.any():
        components, count = label(result, structure=np.ones((3, 3), dtype=np.uint8))
        if count > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            result = components == int(np.argmax(sizes))
    if fill_holes and result.any():
        result = binary_fill_holes(result)
    return np.asarray(result, dtype=bool)


def binary_masks_from_probabilities(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    keep_largest_component: bool = False,
    fill_holes: bool = False,
) -> np.ndarray:
    """Threshold an N x H x W probability batch and apply the frozen postprocess policy."""
    threshold = validate_threshold(threshold)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2:
        probabilities = probabilities[None]
    if probabilities.ndim != 3:
        raise ValueError(f"Expected probabilities shaped N x H x W, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Binary probabilities contain NaN or infinity")
    if probabilities.size and (probabilities.min() < 0.0 or probabilities.max() > 1.0):
        raise ValueError("Binary probabilities must be within [0, 1]")
    masks = probabilities >= threshold
    return np.stack([
        postprocess_binary_mask(mask, keep_largest_component, fill_holes)
        for mask in masks
    ])


def binary_masks_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
    keep_largest_component: bool = False,
    fill_holes: bool = False,
) -> np.ndarray:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"Expected binary logits shaped N x 1 x H x W, got {tuple(logits.shape)}")
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
    return binary_masks_from_probabilities(
        probabilities,
        threshold=threshold,
        keep_largest_component=keep_largest_component,
        fill_holes=fill_holes,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_inference_options(
    config: dict,
    policy_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> tuple[float, bool, bool]:
    if policy_path is not None:
        policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        if int(policy.get("schema_version", 0)) != 1:
            raise RuntimeError(f"Unsupported inference-policy schema in {policy_path}")
        if policy.get("selection_split") != "validation":
            raise RuntimeError("Inference policy must be selected on validation patients only")
        if policy.get("config_sha256") != config_sha256(config):
            raise RuntimeError("Inference-policy config hash does not match the supplied configuration")
        if checkpoint is not None:
            expected = policy.get("checkpoint_sha256")
            actual = sha256_file(checkpoint)
            if expected != actual:
                raise RuntimeError("Inference-policy checkpoint hash does not match the supplied checkpoint")
            manifest_path = run_split_manifest_path(checkpoint)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Frozen split manifest not found next to checkpoint: {manifest_path}")
            expected_manifest = policy.get("split_manifest_sha256")
            actual_manifest = sha256_file(manifest_path)
            if expected_manifest != actual_manifest:
                raise RuntimeError("Inference-policy split-manifest hash does not match the supplied run")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if patient_ids_sha256(manifest.get("val", [])) != policy.get("validation_patient_ids_sha256"):
                raise RuntimeError("Inference-policy validation-patient hash does not match the supplied run")
        return (
            validate_threshold(policy["best_threshold"]),
            bool(policy.get("keep_largest_component", False)),
            bool(policy.get("fill_holes", False)),
        )
    ecfg = config.get("evaluation", {})
    return (
        validate_threshold(ecfg.get("threshold", 0.5)),
        bool(ecfg.get("keep_largest_component", False)),
        bool(ecfg.get("fill_holes", False)),
    )
