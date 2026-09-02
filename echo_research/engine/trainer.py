from __future__ import annotations

import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..losses import BoundaryDiceLoss, LVFunctionalConsistencyLoss, SegmentationLoss
from ..inference import binary_inference_options, binary_masks_from_logits
from ..metrics.segmentation import segmentation_metrics
from .checkpoint import restore_checkpoint, save_checkpoint, save_json


class Trainer:
    def __init__(self, model, config: Dict, train_loader, val_loader, device: torch.device, run_dir: str | Path):
        self.model = model.to(device)
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tcfg = config["training"]
        lcfg = config["loss"]
        self.task = config["data"]["task"]
        self.num_classes = int(config["model"].get("num_classes", 4))
        self.paired = bool(config["data"].get("paired_phases", True))

        self.seg_loss = SegmentationLoss(
            self.task,
            self.num_classes,
            lcfg.get("ce_weight", 1.0),
            lcfg.get("dice_weight", 1.0),
            lcfg.get("bce_pos_weight", 1.0),
        )
        self.boundary_loss = BoundaryDiceLoss(self.task, self.num_classes, lcfg.get("boundary_tolerance_px", 2))
        self.functional_loss = LVFunctionalConsistencyLoss(self.task, lv_class=lcfg.get("lv_class", 1), physiology_weight=lcfg.get("physiology_weight", 0.25))
        self.adv_bce = nn.BCEWithLogitsLoss()

        self.gen_opt = torch.optim.Adam(self.model.generator.parameters(), lr=float(tcfg["lr_g"]), betas=tuple(tcfg.get("betas", [0.5, 0.999])))
        self.disc_opt = torch.optim.Adam(self.model.discriminator.parameters(), lr=float(tcfg["lr_d"]), betas=tuple(tcfg.get("betas", [0.5, 0.999])))
        self.gen_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.gen_opt, mode="max", factor=float(tcfg.get("lr_factor", 0.5)), patience=int(tcfg.get("lr_patience", 10)), min_lr=float(tcfg.get("min_lr", 1e-6)))
        self.disc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.disc_opt, mode="max", factor=float(tcfg.get("lr_factor", 0.5)), patience=int(tcfg.get("lr_patience", 10)), min_lr=float(tcfg.get("min_lr", 1e-6)))
        amp_enabled = bool(tcfg.get("amp", True) and device.type == "cuda")
        if hasattr(torch.amp, "GradScaler"):
            self.scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            self.scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        else:  # PyTorch 2.2 compatibility
            self.scaler_g = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            self.scaler_d = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.history = []
        self.global_step = 0
        self.eval_threshold, self.keep_largest_component, self.fill_holes = binary_inference_options(config)

    def _disc_labels(self, out: torch.Tensor, real: bool) -> torch.Tensor:
        smoothing = float(self.cfg["training"].get("label_smoothing", 0.0))
        if real:
            val = 1.0 - smoothing
        else:
            val = 0.0
        labels = torch.full_like(out, val)
        flip_prob = float(self.cfg["training"].get("label_flip_prob", 0.0))
        if self.model.discriminator.training and flip_prob > 0:
            flip = torch.rand_like(labels) < flip_prob
            labels = torch.where(flip, 1.0 - labels, labels)
        return labels

    def _adversarial_weight(self, epoch: int) -> float:
        base_weight = float(self.cfg["loss"].get("adversarial_weight", 0.0))
        warmup_epochs = int(self.cfg["training"].get("adversarial_warmup_epochs", 0))
        if base_weight <= 0 or epoch <= warmup_epochs:
            return 0.0
        ramp_epochs = int(self.cfg["training"].get("adversarial_ramp_epochs", 0))
        if ramp_epochs <= 0:
            return base_weight
        progress = min(1.0, float(epoch - warmup_epochs) / float(ramp_epochs))
        return base_weight * progress

    def _one_phase_generator_loss(self, image, target, logits, adversarial_weight: float):
        lcfg = self.cfg["loss"]
        total = torch.zeros((), device=self.device)
        details = {}
        seg, seg_parts = self.seg_loss(logits, target)
        total = total + float(lcfg.get("region_weight", 1.0)) * seg
        details.update(seg_parts)

        if float(lcfg.get("boundary_weight", 0.0)) > 0:
            b = self.boundary_loss(logits, target)
            total = total + float(lcfg["boundary_weight"]) * b
            details["boundary_loss"] = b.detach()

        if float(lcfg.get("pixel_l1_weight", 0.0)) > 0:
            fake = self.model.mask_representation(logits, is_logits=True)
            real = self.model.mask_representation(target, is_logits=False)
            pix = F.l1_loss(fake, real)
            total = total + float(lcfg["pixel_l1_weight"]) * pix
            details["pixel_l1"] = pix.detach()

        if adversarial_weight > 0:
            fake = self.model.mask_representation(logits, is_logits=True)
            d_fake = self.model.discriminator(image, fake)
            # Generator targets are always clean real labels. D-only smoothing/flipping must
            # not weaken the generator's adversarial objective.
            adv = self.adv_bce(d_fake, torch.ones_like(d_fake))
            total = total + adversarial_weight * adv
            details["adversarial_loss"] = adv.detach()
            details["adversarial_weight"] = logits.new_tensor(adversarial_weight)
        return total, details

    def _discriminator_loss_phase(self, image, target, logits):
        real_mask = self.model.mask_representation(target, is_logits=False)
        fake_mask = self.model.mask_representation(logits.detach(), is_logits=True)
        out_real = self.model.discriminator(image, real_mask)
        out_fake = self.model.discriminator(image, fake_mask)
        loss_real = self.adv_bce(out_real, self._disc_labels(out_real, True))
        loss_fake = self.adv_bce(out_fake, self._disc_labels(out_fake, False))
        return 0.5 * (loss_real + loss_fake)

    def _move(self, batch):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train_epoch(self, epoch: int):
        self.model.train()
        sums = defaultdict(float)
        count = 0
        d_due_count = 0
        d_update_count = 0
        d_loss_sum = 0.0
        max_grad = float(self.cfg["training"].get("grad_clip", 5.0))
        d_min_loss = float(self.cfg["training"].get("d_update_min_loss", -1.0))
        d_update_every = max(1, int(self.cfg["training"].get("d_update_every", 1)))
        adversarial_weight = self._adversarial_weight(epoch)
        for batch in self.train_loader:
            self.global_step += 1
            batch = self._move(batch)
            if self.paired:
                image_ed, target_ed = batch["image_ed"], batch["mask_ed"]
                image_es, target_es = batch["image_es"], batch["mask_es"]
            else:
                image_ed, target_ed = batch["image"], batch["mask"]
                image_es = target_es = None

            # Generator step. Discriminator weights are frozen to avoid accumulating D gradients.
            for p in self.model.discriminator.parameters():
                p.requires_grad_(False)
            # Freezing parameters does not freeze BatchNorm buffers. Evaluation mode prevents
            # the G-only forward pass from mutating discriminator running statistics.
            self.model.discriminator.eval()
            self.gen_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                logits_ed = self.model.generator(image_ed)
                g_loss, parts = self._one_phase_generator_loss(image_ed, target_ed, logits_ed, adversarial_weight)
                if self.paired:
                    logits_es = self.model.generator(image_es)
                    g2, parts2 = self._one_phase_generator_loss(image_es, target_es, logits_es, adversarial_weight)
                    g_loss = 0.5 * (g_loss + g2)
                    for k, v in parts2.items():
                        parts[k] = 0.5 * (parts.get(k, v) + v)
                    fw = float(self.cfg["loss"].get("functional_weight", 0.0))
                    if fw > 0:
                        fl, fparts = self.functional_loss(logits_ed, logits_es, target_ed, target_es)
                        g_loss = g_loss + fw * fl
                        parts.update(fparts)
                else:
                    logits_es = None
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.unscale_(self.gen_opt)
            torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_grad)
            self.scaler_g.step(self.gen_opt)
            self.scaler_g.update()

            # Discriminator step.
            for p in self.model.discriminator.parameters():
                p.requires_grad_(True)
            self.model.discriminator.train()
            self.disc_opt.zero_grad(set_to_none=True)
            d_due = adversarial_weight > 0 and self.global_step % d_update_every == 0
            d_loss = torch.zeros((), device=self.device)
            update_d = False
            if d_due:
                d_due_count += image_ed.shape[0]
                # Probe a loss floor without changing BatchNorm buffers. If an update is
                # due, recompute in train mode so D receives a normal training forward.
                if d_min_loss >= 0:
                    self.model.discriminator.eval()
                    with torch.no_grad(), torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.amp_enabled,
                    ):
                        probe_loss = self._discriminator_loss_phase(image_ed, target_ed, logits_ed)
                        if self.paired:
                            probe_loss = 0.5 * (
                                probe_loss + self._discriminator_loss_phase(image_es, target_es, logits_es)
                            )
                    update_d = float(probe_loss) >= d_min_loss
                    d_loss = probe_loss
                else:
                    update_d = True
                if update_d:
                    self.model.discriminator.train()
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.amp_enabled,
                    ):
                        d_loss = self._discriminator_loss_phase(image_ed, target_ed, logits_ed)
                        if self.paired:
                            d_loss = 0.5 * (
                                d_loss + self._discriminator_loss_phase(image_es, target_es, logits_es)
                            )
                    self.scaler_d.scale(d_loss).backward()
                    self.scaler_d.unscale_(self.disc_opt)
                    torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_grad)
                    self.scaler_d.step(self.disc_opt)
                    self.scaler_d.update()
                    d_update_count += image_ed.shape[0]
                d_loss_sum += float(d_loss.detach()) * image_ed.shape[0]

            bs = image_ed.shape[0]
            count += bs
            sums["g_loss"] += float(g_loss.detach()) * bs
            for k, v in parts.items():
                sums[k] += float(v) * bs
        stats = {k: v / max(1, count) for k, v in sums.items()}
        stats["d_loss"] = d_loss_sum / max(1, d_due_count)
        stats["d_due_fraction"] = d_due_count / max(1, count)
        stats["d_update_fraction_due"] = d_update_count / max(1, d_due_count)
        # Backward-compatible field: fraction of all samples that triggered a D step.
        stats["d_updated"] = d_update_count / max(1, count)
        return stats

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        dice_values = []
        dice_by_patient = defaultdict(list)
        losses = []
        for batch in self.val_loader:
            batch = self._move(batch)
            if self.paired:
                phases = [(batch["image_ed"], batch["mask_ed"]), (batch["image_es"], batch["mask_es"])]
            else:
                phases = [(batch["image"], batch["mask"])]
            for image, target in phases:
                logits = self.model.generator(image)
                seg, _ = self.seg_loss(logits, target)
                losses.append(float(seg))
                if self.task == "binary":
                    pred = binary_masks_from_logits(
                        logits,
                        threshold=self.eval_threshold,
                        keep_largest_component=self.keep_largest_component,
                        fill_holes=self.fill_holes,
                    )
                    gt = (target[:, 0] if target.ndim == 4 else target).cpu().numpy()
                    for i, (p, t) in enumerate(zip(pred, gt)):
                        score = segmentation_metrics(p, t)["dice"]
                        dice_values.append(score)
                        dice_by_patient[str(batch["patient_id"][i])].append(score)
                else:
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    gt = target.cpu().numpy()
                    # Validation checkpoint metric is macro foreground Dice.
                    for i, (p, t) in enumerate(zip(pred, gt)):
                        per = []
                        for cls in range(1, self.num_classes):
                            per.append(segmentation_metrics(p == cls, t == cls)["dice"])
                        score = float(np.mean(per))
                        dice_values.append(score)
                        dice_by_patient[str(batch["patient_id"][i])].append(score)
        patient_scores = [float(np.mean(values)) for values in dice_by_patient.values()]
        return {
            "val_dice": float(np.mean(patient_scores)),
            "val_dice_frame_mean": float(np.mean(dice_values)),
            "val_patient_count": len(patient_scores),
            "val_region_loss": float(np.mean(losses)),
        }

    def _save_training_checkpoint(self, path: Path, epoch: int, best: float, no_improve: int) -> None:
        save_checkpoint(
            path,
            self.model,
            self.gen_opt,
            self.disc_opt,
            epoch,
            best,
            self.cfg,
            gen_sched=self.gen_sched,
            disc_sched=self.disc_sched,
            scaler_g=self.scaler_g,
            scaler_d=self.scaler_d,
            history=self.history,
            global_step=self.global_step,
            no_improve=no_improve,
        )

    def _check_resume_config(self, checkpoint_config: Dict | None) -> None:
        if not checkpoint_config:
            return
        expected = {
            "task": self.cfg["data"].get("task"),
            "num_classes": self.cfg["model"].get("num_classes", 4),
            "base_channels": self.cfg["model"].get("base_channels", 64),
            "normalization": self.cfg["model"].get("normalization", "batch"),
        }
        found = {
            "task": checkpoint_config.get("data", {}).get("task"),
            "num_classes": checkpoint_config.get("model", {}).get("num_classes", 4),
            "base_channels": checkpoint_config.get("model", {}).get("base_channels", 64),
            "normalization": checkpoint_config.get("model", {}).get("normalization", "batch"),
        }
        if expected != found:
            raise RuntimeError(f"Resume checkpoint architecture mismatch: expected {expected}, found {found}")
        current_data = dict(self.cfg.get("data", {}))
        saved_data = dict(checkpoint_config.get("data", {}))
        current_data.pop("root", None)
        saved_data.pop("root", None)
        current_training = dict(self.cfg.get("training", {}))
        saved_training = dict(checkpoint_config.get("training", {}))
        for mutable_key in ("epochs", "num_workers", "checkpoint_every"):
            current_training.pop(mutable_key, None)
            saved_training.pop(mutable_key, None)
        frozen_sections_match = (
            current_data == saved_data
            and self.cfg.get("model", {}) == checkpoint_config.get("model", {})
            and self.cfg.get("loss", {}) == checkpoint_config.get("loss", {})
            and self.cfg.get("evaluation", {}) == checkpoint_config.get("evaluation", {})
            and current_training == saved_training
        )
        if not frozen_sections_match:
            raise RuntimeError(
                "Resume config changed frozen data/preprocessing, model, loss, evaluation, or training settings. "
                "Start a new run instead of resuming."
            )

    def fit(self, resume_checkpoint: str | Path | None = None):
        tcfg = self.cfg["training"]
        max_epochs = int(tcfg["epochs"])
        patience = int(tcfg.get("early_stopping_patience", 30))
        min_delta = float(tcfg.get("early_stopping_min_delta", 1e-4))
        best = -math.inf
        no_improve = 0
        start_epoch = 1
        if resume_checkpoint is not None:
            ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            self._check_resume_config(ckpt.get("config"))
            restore_checkpoint(
                ckpt,
                self.model,
                self.gen_opt,
                self.disc_opt,
                gen_sched=self.gen_sched,
                disc_sched=self.disc_sched,
                scaler_g=self.scaler_g,
                scaler_d=self.scaler_d,
                restore_rng=True,
            )
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best = float(ckpt.get("best_metric", best))
            no_improve = int(ckpt.get("no_improve", 0))
            self.global_step = int(ckpt.get("global_step", 0))
            self.history = list(ckpt.get("history", []))
            if int(ckpt.get("checkpoint_version", 1)) < 2:
                print("WARNING: legacy checkpoint resumed without exact scheduler/scaler/RNG/history state.")
            print(f"Resuming from epoch {start_epoch} with best validation Dice {best:.4f}.")
            del ckpt
        start = time.time()
        last_completed_epoch = start_epoch - 1
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                train_stats = self.train_epoch(epoch)
                val_stats = self.validate()
                metric = val_stats["val_dice"]
                scheduler_start = int(tcfg.get("lr_scheduler_start_epoch", 1))
                if epoch >= scheduler_start:
                    self.gen_sched.step(metric)
                if (
                    epoch >= scheduler_start
                    and self._adversarial_weight(epoch) > 0
                    and train_stats.get("d_updated", 0.0) > 0
                ):
                    self.disc_sched.step(metric)
                improved = metric > best + min_delta
                if improved:
                    best = metric
                    no_improve = 0
                else:
                    no_improve += 1
                row = {
                    "epoch": epoch,
                    **train_stats,
                    **val_stats,
                    "lr_g": self.gen_opt.param_groups[0]["lr"],
                    "lr_d": self.disc_opt.param_groups[0]["lr"],
                    "best_val_dice": best,
                    "elapsed_sec": time.time() - start,
                }
                self.history.append(row)
                save_json(self.run_dir / "history.json", self.history)
                last_completed_epoch = epoch
                self._save_training_checkpoint(self.run_dir / "last.pt", epoch, best, no_improve)
                if improved:
                    shutil.copy2(self.run_dir / "last.pt", self.run_dir / "best.pt")
                checkpoint_every = int(tcfg.get("checkpoint_every", 0))
                if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                    shutil.copy2(self.run_dir / "last.pt", self.run_dir / f"epoch_{epoch:03d}.pt")
                adv_weight = self._adversarial_weight(epoch)
                completed_this_session = epoch - start_epoch + 1
                eta_seconds = int(
                    max(0.0, (time.time() - start) / max(1, completed_this_session) * (max_epochs - epoch))
                )
                eta_hours, eta_remainder = divmod(eta_seconds, 3600)
                eta_minutes, eta_seconds_only = divmod(eta_remainder, 60)
                eta_text = f"{eta_hours:d}:{eta_minutes:02d}:{eta_seconds_only:02d}"
                print(
                    f"Epoch {epoch:03d} | G {row['g_loss']:.4f} | D {row['d_loss']:.4f} | "
                    f"D due/upd {row['d_due_fraction']:.2f}/{row['d_update_fraction_due']:.2f} | "
                    f"adv_w {adv_weight:.4f} | val Dice {metric:.4f} | best {best:.4f} | ETA {eta_text}"
                )
                if no_improve >= patience:
                    print(f"Early stopping: validation Dice did not improve for {patience} epochs.")
                    break
        except KeyboardInterrupt:
            last_path = self.run_dir / "last.pt"
            if last_path.exists():
                shutil.copy2(last_path, self.run_dir / "interrupted.pt")
                print(f"Training interrupted. Recoverable checkpoint saved at {self.run_dir / 'interrupted.pt'}.")
            else:
                self._save_training_checkpoint(self.run_dir / "interrupted.pt", 0, best, no_improve)
                print(
                    "Training interrupted during the first epoch. Saved partial weights at interrupted.pt; "
                    "resuming restarts epoch 1 and is not a bitwise continuation of the interrupted batch."
                )
            raise
        if last_completed_epoch < 1:
            raise RuntimeError(f"No epochs were run: resume epoch {start_epoch} exceeds configured maximum {max_epochs}")
        return best
