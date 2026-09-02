from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def save_checkpoint(
    path: str | Path,
    model,
    gen_opt,
    disc_opt,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    *,
    gen_sched=None,
    disc_sched=None,
    scaler_g=None,
    scaler_d=None,
    history=None,
    global_step: int = 0,
    no_improve: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 2,
        "epoch": epoch,
        "best_metric": best_metric,
        "generator": model.generator.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "gen_optimizer": gen_opt.state_dict() if gen_opt else None,
        "disc_optimizer": disc_opt.state_dict() if disc_opt else None,
        "gen_scheduler": gen_sched.state_dict() if gen_sched else None,
        "disc_scheduler": disc_sched.state_dict() if disc_sched else None,
        "scaler_g": scaler_g.state_dict() if scaler_g else None,
        "scaler_d": scaler_d.state_dict() if scaler_d else None,
        "history": list(history or []),
        "global_step": int(global_step),
        "no_improve": int(no_improve),
        "rng_state": capture_rng_state(),
        "config": config,
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def load_checkpoint(
    path: str | Path,
    model,
    gen_opt=None,
    disc_opt=None,
    map_location="cpu",
    *,
    gen_sched=None,
    disc_sched=None,
    scaler_g=None,
    scaler_d=None,
    restore_rng: bool = False,
):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    restore_checkpoint(
        ckpt,
        model,
        gen_opt,
        disc_opt,
        gen_sched=gen_sched,
        disc_sched=disc_sched,
        scaler_g=scaler_g,
        scaler_d=scaler_d,
        restore_rng=restore_rng,
    )
    return ckpt


def restore_checkpoint(
    ckpt: Dict[str, Any],
    model,
    gen_opt=None,
    disc_opt=None,
    *,
    gen_sched=None,
    disc_sched=None,
    scaler_g=None,
    scaler_d=None,
    restore_rng: bool = False,
) -> None:
    """Restore an already loaded payload without reading a large checkpoint twice."""
    model.generator.load_state_dict(ckpt["generator"])
    model.discriminator.load_state_dict(ckpt["discriminator"])
    if gen_opt is not None and ckpt.get("gen_optimizer"):
        gen_opt.load_state_dict(ckpt["gen_optimizer"])
    if disc_opt is not None and ckpt.get("disc_optimizer"):
        disc_opt.load_state_dict(ckpt["disc_optimizer"])
    if gen_sched is not None and ckpt.get("gen_scheduler"):
        gen_sched.load_state_dict(ckpt["gen_scheduler"])
    if disc_sched is not None and ckpt.get("disc_scheduler"):
        disc_sched.load_state_dict(ckpt["disc_scheduler"])
    if scaler_g is not None and ckpt.get("scaler_g"):
        scaler_g.load_state_dict(ckpt["scaler_g"])
    if scaler_d is not None and ckpt.get("scaler_d"):
        scaler_d.load_state_dict(ckpt["scaler_d"])
    if restore_rng:
        restore_rng_state(ckpt.get("rng_state"))


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
