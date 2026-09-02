# Validation Report

## High-Dice recovery addendum — 24 August 2026

- All 60 Python source/test files after the high-Dice recovery changes: bytecode compile
  and independent AST parse **PASS**.
- Three independent static reviews covered training dynamics, data/metric leakage, and
  PyTorch compatibility; their blocking findings were corrected.
- New regression coverage was added for inference-policy hashes, 8-connected
  postprocessing, probability/range validation, augmentation synchronization,
  high-Dice configuration controls, and full checkpoint optimizer/scheduler/RNG restore.
- Runtime pytest and numerical training were **not rerun in this Windows workspace**
  because its bundled Python does not include PyTorch, SciPy, PyYAML, or pytest. Run
  `python scripts/smoke_test.py && pytest` in the installed Linux/A4000 environment
  before the long CAMUS experiment.

The checks below are the original 11 August build validation. They remain useful
historical evidence, but do not substitute for executing the expanded suite after the
24 August changes.

**Validation date:** 11 August 2026

## Checks performed in the build environment

- Python bytecode compilation for the new package, scripts, and tests: **PASS**.
- Pytest suite: **10 tests PASS**.
- Binary U-Net/PatchGAN forward pass: **PASS**.
- Multiclass U-Net/PatchGAN forward pass: **PASS**.
- Binary segmentation + boundary + functional losses: forward/backward **PASS**.
- Multiclass segmentation + boundary + functional losses: forward/backward **PASS**.
- One complete synthetic paired ED/ES binary trainer epoch including generator update, discriminator update, validation, checkpointing, and early-stopping state: **PASS**.
- One complete synthetic paired ED/ES multiclass trainer epoch: **PASS**.
- Automatic split tests verify patient-disjoint train/validation/test behavior: **PASS**.
- Explicit test split exclusion from auto-derived validation: **PASS**.
- Editable package installation with local build isolation disabled: **PASS**.

## Important things that cannot be truthfully validated without the real dataset/GPU

The conversation supplied project source/context and the reference paper, not the CAMUS image files themselves. Therefore the following still require the user's actual CAMUS installation:

1. exact compatibility with the local CAMUS directory variant;
2. real NIfTI metadata/spacing values;
3. full A4000/Kaggle VRAM usage and training speed;
4. final Dice/HD95/ASD numbers;
5. clinical EDV/ESV/EF validity;
6. cross-domain performance on EchoNet-Dynamic.

The repository includes `scripts/check_dataset.py` specifically to validate the first three data assumptions before a long training run.

## Clinical-volume warning

`echo_research/metrics/clinical.py` contains an experimental biplane Simpson-style estimator so the full research workflow is scaffolded. It is intentionally marked **not publication-validated**. Before any paper reports mL or EF from it, compare ground-truth-mask-derived values against trusted CAMUS clinical references or replace it with a validated/official implementation.

## Legacy code

`legacy/exact_source/` is preserved from the uploaded project for traceability. It is not presented as the corrected research pipeline. The corrected pipeline is `echo_research/` + `scripts/` + `configs/`.
