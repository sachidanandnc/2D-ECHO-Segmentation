#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from echo_research.config import load_config
from echo_research.engine.trainer import Trainer
from echo_research.factory import build_loaders, build_model
from echo_research.run_manifest import verify_run_split
from echo_research.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="Override data.root without editing YAML")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=None, help="Override experiment.seed; recorded in resolved config")
    ap.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size; useful if GPU memory is limited")
    ap.add_argument(
        "--resume",
        default=None,
        help="Resume from the next epoch using a v2 last.pt/interrupted.pt checkpoint",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    seed = int(cfg["experiment"].get("seed", 2026))
    seed_everything(seed, bool(cfg["experiment"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = cfg["experiment"].get("name", Path(args.config).stem)
    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        if not checkpoint_path.is_file():
            raise SystemExit(f"Resume checkpoint does not exist: {checkpoint_path}")
        checkpoint_run_dir = checkpoint_path.parent
        if args.run_dir is not None and Path(args.run_dir).resolve() != checkpoint_run_dir:
            raise SystemExit(
                "--run-dir must be the checkpoint's parent directory when resuming. "
                "Start a new run instead if you intend to change the experiment."
            )
        run_dir = checkpoint_run_dir
    else:
        run_dir = Path(args.run_dir or Path("runs") / name / f"seed_{seed}")
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(
                f"Refusing to overwrite non-empty run directory: {run_dir.resolve()}\n"
                "Choose a new --run-dir, or use --resume with that run's last.pt."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, splits = build_loaders(cfg)
    split_manifest = {"train": splits[0], "val": splits[1], "test": splits[2]}
    if args.resume:
        for split_name, ids in split_manifest.items():
            verify_run_split(args.resume, split_name, ids, required=True)
        (run_dir / "resolved_config_resume.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    else:
        (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    model = build_model(cfg)
    n_g = sum(p.numel() for p in model.generator.parameters())
    n_d = sum(p.numel() for p in model.discriminator.parameters())
    print(f"Device: {device} | Generator params: {n_g:,} | Discriminator params: {n_d:,}")
    Trainer(model, cfg, train_loader, val_loader, device, run_dir).fit(args.resume)
    print(f"Run saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
