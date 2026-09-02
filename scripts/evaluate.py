#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from echo_research.config import load_config
from echo_research.engine.evaluator import evaluate_checkpoint
from echo_research.factory import build_model, build_test_loader, split_ids_from_config
from echo_research.inference import binary_inference_options
from echo_research.run_manifest import verify_run_split


def main():
    ap = argparse.ArgumentParser(description="Final TEST evaluation. Do not use this for model selection.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output", default="test_metrics.csv")
    ap.add_argument("--threshold", type=float, default=None, help="Frozen validation-selected threshold")
    ap.add_argument("--inference-policy", default=None, help="Validation calibration JSON from tune_threshold.py")
    args = ap.parse_args()
    if args.threshold is not None and args.inference_policy is not None:
        raise SystemExit("Use either --threshold or --inference-policy, not both.")
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if cfg.get("evaluation", {}).get("require_calibration_policy", False) and args.inference_policy is None:
        raise SystemExit("This experiment requires --inference-policy from validation-only tune_threshold.py.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    _, _, test_ids = split_ids_from_config(cfg)
    verify_run_split(args.checkpoint, "test", test_ids, required=True)
    loader = build_test_loader(cfg)
    threshold, keep_largest, fill_holes = binary_inference_options(cfg, args.inference_policy, args.checkpoint)
    if args.threshold is not None:
        threshold = args.threshold
    df, summary, patient_summary = evaluate_checkpoint(
        model,
        loader,
        args.checkpoint,
        device,
        cfg["data"]["task"],
        int(cfg["model"].get("num_classes", 4)),
        args.output,
        threshold=threshold,
        keep_largest_component=keep_largest,
        fill_holes=fill_holes,
        original_grid=bool(cfg.get("evaluation", {}).get("original_grid", False)),
        structure=cfg["data"].get("structure", "LVendo"),
    )
    print("Frame-stratified summary:")
    print(summary)
    print("Patient-level primary summary:")
    print(patient_summary)
    print(f"Inference policy: threshold={threshold:.4f}, keep_largest_component={keep_largest}, fill_holes={fill_holes}")
    print(f"Per-sample metrics: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
