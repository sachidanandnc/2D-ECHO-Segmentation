#!/usr/bin/env python3
"""Select one binary threshold on validation patients only, then freeze it for test."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from echo_research.config import load_config
from echo_research.data.camus import LABELS, load_nifti
from echo_research.engine.checkpoint import load_checkpoint
from echo_research.factory import build_model, build_validation_loader, split_ids_from_config
from echo_research.inference import binary_masks_from_probabilities, sha256_file
from echo_research.metrics.segmentation import segmentation_metrics
from echo_research.run_manifest import config_sha256, patient_ids_sha256, verify_run_split


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Validation-only threshold calibration; never run this on test data.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output", default="threshold_calibration.json")
    ap.add_argument("--min-threshold", type=float, default=0.20)
    ap.add_argument("--max-threshold", type=float, default=0.80)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")
    if not 0.0 <= args.min_threshold <= args.max_threshold <= 1.0:
        raise SystemExit("Threshold search bounds must satisfy 0 <= min <= max <= 1")

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if cfg["data"].get("task") != "binary":
        raise SystemExit("Threshold calibration currently supports the binary LV track only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    _, validation_ids, _ = split_ids_from_config(cfg)
    manifest_path, manifest_hash = verify_run_split(args.checkpoint, "val", validation_ids, required=True)
    checkpoint_payload = load_checkpoint(args.checkpoint, model, map_location=device)
    saved_config = checkpoint_payload.get("config")
    if not saved_config:
        raise RuntimeError("Threshold calibration requires a v2 checkpoint containing its resolved config")
    if config_sha256(saved_config) != config_sha256(cfg):
        raise RuntimeError(
            "Calibration config does not match the checkpoint's frozen data/model/evaluation config "
            "(the data root and training batch mechanics may differ)."
        )
    del checkpoint_payload
    model.eval()
    loader = build_validation_loader(cfg)
    ecfg = cfg.get("evaluation", {})
    original_grid = bool(ecfg.get("original_grid", False))
    probabilities, targets, patient_ids = [], [], []
    for batch in loader:
        logits = model.generator(batch["image"].to(device))
        probability_batch = torch.sigmoid(logits).cpu()
        target_batch = batch["mask"]
        for i, patient_id in enumerate(batch["patient_id"]):
            if original_grid:
                original_shape = tuple(int(x) for x in batch["original_shape"][i].numpy())
                probability = F.interpolate(
                    probability_batch[i:i + 1],
                    size=original_shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                gt, _ = load_nifti(Path(batch["mask_path"][i]))
                target = gt == LABELS[cfg["data"].get("structure", "LVendo")]
            else:
                probability = probability_batch[i, 0].numpy()
                target = (target_batch[i, 0] if target_batch.ndim == 4 else target_batch[i]).numpy()
            probabilities.append(probability)
            targets.append(target)
            patient_ids.append(str(patient_id))

    keep_largest = bool(ecfg.get("keep_largest_component", False))
    fill_holes = bool(ecfg.get("fill_holes", False))
    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.steps)

    def patient_mean_dice(threshold: float) -> float:
        grouped = defaultdict(list)
        for probability, target, patient_id in zip(probabilities, targets, patient_ids):
            prediction = binary_masks_from_probabilities(
                probability,
                threshold,
                keep_largest_component=keep_largest,
                fill_holes=fill_holes,
            )[0]
            grouped[patient_id].append(segmentation_metrics(prediction, target)["dice"])
        patient_scores = [float(np.mean(values)) for values in grouped.values()]
        return float(np.mean(patient_scores))

    curve = [
        {"threshold": float(threshold), "mean_val_dice": patient_mean_dice(float(threshold))}
        for threshold in thresholds
    ]
    # Stable tie-break: choose the threshold closest to 0.5, then the lower threshold.
    best = sorted(curve, key=lambda row: (-row["mean_val_dice"], abs(row["threshold"] - 0.5), row["threshold"]))[0]
    default_score = patient_mean_dice(0.5)
    result = {
        "schema_version": 1,
        "selection_split": "validation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": config_sha256(cfg),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": manifest_hash,
        "validation_patient_ids_sha256": patient_ids_sha256(validation_ids),
        "validation_patient_count": len(set(patient_ids)),
        "validation_frame_count": len(patient_ids),
        "best_threshold": best["threshold"],
        "best_mean_val_dice": best["mean_val_dice"],
        "default_0_5_mean_val_dice": default_score,
        "gain_over_default_0_5": best["mean_val_dice"] - default_score,
        "evaluation_grid": "original" if original_grid else "network",
        "keep_largest_component": keep_largest,
        "fill_holes": fill_holes,
        "curve": curve,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    print(f"Use the frozen policy for test: scripts/evaluate.py --inference-policy {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
