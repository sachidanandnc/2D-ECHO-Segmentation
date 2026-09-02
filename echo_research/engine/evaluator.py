from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.camus import LABELS, load_nifti
from ..inference import binary_masks_from_logits, binary_masks_from_probabilities
from ..metrics.segmentation import segmentation_metrics
from .checkpoint import load_checkpoint


@torch.no_grad()
def evaluate_checkpoint(
    model,
    loader,
    checkpoint: str | Path,
    device: torch.device,
    task: str,
    num_classes: int,
    output_csv: str | Path,
    *,
    threshold: float = 0.5,
    keep_largest_component: bool = False,
    fill_holes: bool = False,
    original_grid: bool = False,
    structure: str = "LVendo",
):
    if original_grid and task != "binary":
        raise NotImplementedError("Original-grid evaluation is currently implemented for the binary track only")
    if task == "binary" and structure not in LABELS:
        raise ValueError(f"Unknown binary structure {structure!r}")
    load_checkpoint(checkpoint, model, map_location=device)
    model.to(device).eval()
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"]
        logits = model.generator(image)
        if task == "binary":
            if original_grid:
                probabilities = torch.sigmoid(logits).cpu()
            else:
                pred = binary_masks_from_logits(
                    logits,
                    threshold=threshold,
                    keep_largest_component=keep_largest_component,
                    fill_holes=fill_holes,
                )
                gt = (target[:, 0] if target.ndim == 4 else target).numpy()
            classes = [("foreground", 1)]
        else:
            labels = torch.argmax(logits, dim=1).cpu().numpy()
            gt_labels = target.numpy()
            classes = [(f"class_{c}", c) for c in range(1, num_classes)]
        for i in range(image.shape[0]):
            if task == "binary" and original_grid:
                original_shape = tuple(int(x) for x in batch["original_shape"][i].numpy())
                probability = F.interpolate(
                    probabilities[i:i + 1],
                    size=original_shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                pred_i = binary_masks_from_probabilities(
                    probability,
                    threshold=threshold,
                    keep_largest_component=keep_largest_component,
                    fill_holes=fill_holes,
                )[0]
                gt_original, _ = load_nifti(Path(batch["mask_path"][i]))
                gt_i = gt_original == LABELS[structure]
                spacing = tuple(float(x) for x in batch["original_spacing"][i].numpy())
            else:
                spacing = tuple(float(x) for x in batch["spacing"][i].numpy())
            base = {
                "patient_id": batch["patient_id"][i],
                "view": batch["view"][i],
                "phase": batch["phase"][i],
                "evaluation_grid": "original" if original_grid else "network",
            }
            if task == "binary":
                m = (
                    segmentation_metrics(pred_i, gt_i, spacing)
                    if original_grid
                    else segmentation_metrics(pred[i], gt[i], spacing)
                )
                pred_mask = pred_i if original_grid else pred[i]
                target_mask = gt_i if original_grid else gt[i]
                rows.append({
                    **base,
                    "class": "foreground",
                    **m,
                    "pred_empty": int(not np.asarray(pred_mask).any()),
                    "target_empty": int(not np.asarray(target_mask).any()),
                })
            else:
                for name, c in classes:
                    pred_mask = labels[i] == c
                    target_mask = gt_labels[i] == c
                    m = segmentation_metrics(pred_mask, target_mask, spacing)
                    rows.append({
                        **base,
                        "class": name,
                        **m,
                        "pred_empty": int(not pred_mask.any()),
                        "target_empty": int(not target_mask.any()),
                    })
    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    metric_columns = ["dice", "iou", "precision", "recall", "hd95", "asd", "pred_empty", "target_empty"]
    summary = df.groupby(["view", "phase", "class"])[metric_columns].agg(["mean", "std"])
    summary.to_csv(Path(output_csv).with_name(Path(output_csv).stem + "_summary.csv"))
    patient_metrics = df.groupby(["patient_id", "class"], as_index=False)[metric_columns].mean()
    patient_summary = patient_metrics.groupby("class")[metric_columns].agg(["mean", "std"])
    patient_summary.to_csv(Path(output_csv).with_name(Path(output_csv).stem + "_patient_summary.csv"))
    return df, summary, patient_summary
