# High-Dice Recovery Changes

This update addresses the training pattern shown in the supplied screenshots, where
validation Dice peaked at 0.8096 while discriminator loss collapsed and generator loss
rose.

## Training changes

- Added a direct BCE + soft-Dice control profile: `configs/high_dice_region_only.yaml`.
- Added a conservative comparison profile: `configs/high_dice_binary.yaml`.
- Added InstanceNorm and Pix2Pix weight initialization options.
- Added synchronized ED/ES affine and intensity augmentation.
- Added positive-class weighting, boundary, and functional loss controls.
- Added GAN warm-up/ramp, clean generator targets, discriminator cadence and loss-floor
  throttling, and discriminator BatchNorm-buffer isolation.
- Corrected discriminator logging to average only due steps and separately report due
  and update fractions.
- Delayed LR plateau scheduling until the GAN ramp is complete in the comparison run.

## Integrity and evaluation changes

- Added atomic full-state `last.pt`, `best.pt`, periodic, and interruption checkpoints.
- Added optimizer, scheduler, AMP scaler, history, global-step, and RNG restoration.
- Added run-directory overwrite protection and strict resume config/split checks.
- Added validation-only patient-level threshold calibration.
- Changed checkpoint selection to mean patient Dice while retaining frame-mean Dice in
  history for auditability.
- Bound calibration policies to exact checkpoint and split hashes.
- Centralized threshold, largest-component, and hole-filling behavior across validation,
  test, robustness, and clinical evaluation.
- Calibrated and final test masks are thresholded on the original NIfTI grid; clinical
  geometry now upsamples probabilities before thresholding.

## Verification status

The source tree has been statically parsed and compiled. The local Windows workspace
does not contain the CAMUS dataset or a PyTorch runtime, so numerical training and the
full pytest suite must be run in the Linux/A4000 environment shown in the screenshots.
No code change can promise a 0.90 Dice before that frozen validation run is completed.
