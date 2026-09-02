# Echo Research Paper Kit

A research-oriented continuation of the college major project that reproduced the 2024 IEEE TUFFC Pix2Pix-GAN echocardiography segmentation paper.

The ZIP contains **two things on purpose**:

1. `legacy/` preserves the supplied TensorFlow project and its context exactly enough to return to it later.
2. `echo_research/` is a clean PyTorch research framework designed for reproducible experiments on an NVIDIA A4000 or Kaggle GPU.

The primary paper track is intentionally focused on **LV endocardium segmentation** first. It extends Pix2Pix with:

- strict patient-level train/validation/test isolation;
- region supervision (BCE + Dice);
- boundary-aware supervision;
- paired ED/ES functional-consistency supervision;
- robustness experiments;
- patient-level clinical evaluation hooks;
- ablations and five-seed experiments;
- reproducibility manifests and paper templates.

> Important: this kit is designed to make a publishable study *possible*. It cannot guarantee acceptance or guarantee that a novelty claim is unique. The novelty gate in `paper/CLAIMS_AND_EVIDENCE.md` must be completed immediately before submission because the literature changes.

## First run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python scripts/smoke_test.py
pytest
```

Check CAMUS before training:

```bash
python scripts/check_dataset.py \
  --config configs/proposed_binary.yaml \
  --data-root /path/to/CAMUS_public
```

Train the proposed model:

```bash
python scripts/train.py \
  --config configs/proposed_binary.yaml \
  --data-root /path/to/CAMUS_public
```

## High-Dice recovery run

`configs/baseline_binary.yaml` is an intentionally historical L1 + GAN baseline. It is
not the performance configuration. If that baseline peaks early near 0.81 Dice while
the discriminator loss collapses, do not continue that checkpoint. First establish the
stable region-only control:

```bash
python scripts/check_dataset.py \
  --config configs/high_dice_region_only.yaml \
  --data-root /path/to/CAMUS_public

python scripts/train.py \
  --config configs/high_dice_region_only.yaml \
  --data-root /path/to/CAMUS_public
```

Then run `configs/high_dice_binary.yaml` as the small-auxiliary/GAN comparison. Both
profiles use BCE + Dice as the dominant objective, InstanceNorm for batch-size-one/two
stability, augmentation, and a recoverable checkpoint after every completed epoch. The
full profile adds a 15-epoch segmentation warm-up, a small ramped GAN term, and
discriminator throttling. If batch size 2 does not fit, append `--batch-size 1` and use a
new run directory for that changed batch-size experiment.

Resume the same run after an interruption (do not resume a baseline checkpoint into a
different architecture/configuration):

```bash
python scripts/train.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --run-dir runs/high_dice_binary_lv/seed_2026 \
  --resume runs/high_dice_binary_lv/seed_2026/last.pt
```

After model/loss choices are frozen, select one threshold on validation only:

```bash
python scripts/tune_threshold.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --checkpoint runs/high_dice_binary_lv/seed_2026/best.pt \
  --output runs/high_dice_binary_lv/seed_2026/threshold_calibration.json
```

Then use the generated policy unchanged for the one-time test evaluation:

```bash
python scripts/evaluate.py \
  --config configs/high_dice_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --checkpoint runs/high_dice_binary_lv/seed_2026/best.pt \
  --inference-policy runs/high_dice_binary_lv/seed_2026/threshold_calibration.json \
  --output runs/high_dice_binary_lv/seed_2026/test_metrics.csv
```

The policy records and verifies hashes of the selected checkpoint and frozen validation
split. A 0.90+ validation/test Dice is a target, not a guarantee; it must be established
on the frozen patient split rather than obtained by tuning on test.

Resume is epoch-level recovery. With multi-worker randomized augmentation, a resumed
run is statistically equivalent but not guaranteed to be bit-for-bit identical to an
uninterrupted run.

Final test evaluation only after model choices are frozen:

```bash
python scripts/evaluate.py \
  --config configs/proposed_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --checkpoint runs/proposed_boundary_functional_pix2pix_lv/seed_2026/best.pt \
  --output runs/proposed_boundary_functional_pix2pix_lv/seed_2026/test_metrics.csv
```

For the complete experimental order, read **`CONTEXT.md` first**, then `paper/EXPERIMENT_PROTOCOL.md`.
