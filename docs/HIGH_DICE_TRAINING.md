# High-Dice Training and Recovery Guide

## Why the 0.8096 run degraded

The 200-epoch screenshot matches `configs/baseline_binary.yaml`. That configuration is
the historical reproduction baseline:

- augmentation is disabled;
- BCE, Dice, boundary, and functional losses are disabled;
- the generator uses `100 x L1 + 1 x adversarial loss`;
- generator and discriminator use the same learning rate;
- the discriminator is updated on every batch.

The log pattern (discriminator loss falling toward zero while generator loss rises) is
consistent with discriminator domination. The epoch-3 `best.pt` remains the valid best
baseline checkpoint, but continuing that exact run is unlikely to turn 0.8096 into 0.90.

## What the new profile changes

`configs/high_dice_region_only.yaml` is the stable control and
`configs/high_dice_binary.yaml` is the small-auxiliary/GAN comparison:

1. BCE + soft Dice dominate the objective.
2. The control disables L1, boundary, functional, and GAN terms so performance can be
   attributed to direct segmentation supervision.
3. The comparison keeps those auxiliary terms small; the first 15 epochs are
   segmentation-only and GAN weight ramps in over 20 epochs.
4. The discriminator uses a lower learning rate, updates every second global step, and
   skips updates when its loss is already below 0.30.
5. InstanceNorm removes batch-size-one BatchNorm instability.
6. Moderate shared ED/ES geometry and intensity augmentation improves generalization.
7. Validation uses a deterministic largest-component and hole-filling policy.
8. `last.pt` is atomically saved after every completed epoch; Ctrl+C also preserves a
   recoverable `interrupted.pt` copy.

## Recommended run sequence

Run the region-only control first:

```bash
python scripts/check_dataset.py \
  --config configs/high_dice_region_only.yaml \
  --data-root /path/to/CAMUS_public

python scripts/train.py \
  --config configs/high_dice_region_only.yaml \
  --data-root /path/to/CAMUS_public
```

Then run the auxiliary/GAN comparison in its separate default run directory:

```bash
python scripts/check_dataset.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public

python scripts/train.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public
```

If the A4000 reports out-of-memory:

```bash
python scripts/train.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --batch-size 1
```

If that run is later resumed, include the same `--batch-size 1` override again.

After an interruption, use the same config, data root, seed, and run directory:

```bash
python scripts/train.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --run-dir runs/high_dice_binary_lv/seed_2026 \
  --resume runs/high_dice_binary_lv/seed_2026/last.pt
```

Recovery resumes at the next epoch. Multi-worker random augmentation is not guaranteed
to reproduce the uninterrupted run bit-for-bit, although model, optimizer, scheduler,
AMP-scaler, history, and main-process random state are restored.

## Acceptance gates

Before the full 200-epoch run, confirm:

- the split manifest is identical to the baseline run;
- region Dice loss decreases during the 15-epoch warm-up;
- validation Dice no longer collapses when the GAN starts;
- `d_due_fraction` reflects the configured cadence, `d_update_fraction_due` falls when
  the discriminator is already strong, and due-step discriminator loss does not remain
  near zero;
- predictions contain one plausible LV cavity rather than islands;
- `best.pt`, `last.pt`, `history.json`, `resolved_config.json`, and
  `split_manifest.json` are present.

## Threshold calibration

Run `scripts/tune_threshold.py` only on validation patients. Freeze its reported threshold
and use it unchanged for test:

```bash
python scripts/tune_threshold.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --checkpoint runs/high_dice_binary_lv/seed_2026/best.pt \
  --output runs/high_dice_binary_lv/seed_2026/threshold_calibration.json

python scripts/evaluate.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --checkpoint runs/high_dice_binary_lv/seed_2026/best.pt \
  --inference-policy runs/high_dice_binary_lv/seed_2026/threshold_calibration.json \
  --output runs/high_dice_binary_lv/seed_2026/test_metrics.csv
```

The policy is bound to the checkpoint and validation split by hashes. Never search
thresholds on the test set. Report 0.90+ only if it is achieved on the frozen held-out
patients, and accompany Dice with HD95, ASD, and empty-mask rate. These high-Dice
profiles deliberately refuse final test, robustness, or clinical evaluation without the
validation calibration policy. Calibration and final test threshold probabilities on
the original NIfTI grid before postprocessing.

## Decision rule

- If the region-only control is below 0.85, inspect masks, normalization, split files,
  and prediction overlays before changing loss weights; the pipeline or data is likely
  the limiting factor.
- If it reaches 0.88-0.90, run the auxiliary/GAN comparison once. Keep it only if it
  improves the frozen validation metric and does not worsen HD95/ASD.
- If it reaches at least 0.90, freeze the configuration and calibration policy before
  opening the test result. Do not repeatedly tune on test.
