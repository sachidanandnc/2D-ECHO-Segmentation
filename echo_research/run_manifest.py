from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Sequence


def run_split_manifest_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint).resolve().parent / "split_manifest.json"


def verify_run_split(
    checkpoint: str | Path,
    split_name: str,
    current_ids: Sequence[str],
    *,
    required: bool = True,
) -> tuple[Path | None, str | None]:
    path = run_split_manifest_path(checkpoint)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Frozen run split manifest not found next to checkpoint: {path}")
        return None, None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if split_name not in manifest:
        raise RuntimeError(f"Frozen split manifest has no {split_name!r} split: {path}")
    frozen = sorted(str(item) for item in manifest[split_name])
    current = sorted(str(item) for item in current_ids)
    if frozen != current:
        missing = sorted(set(frozen) - set(current))[:10]
        added = sorted(set(current) - set(frozen))[:10]
        raise RuntimeError(
            f"Current {split_name} patients do not match the frozen run manifest. "
            f"Missing={missing}, added={added}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def patient_ids_sha256(patient_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(item) for item in patient_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_sha256(config: dict) -> str:
    """Hash inference-relevant config while allowing paths/batch mechanics to relocate."""
    source = deepcopy(config)
    experiment = source.get("experiment", {})
    experiment.pop("config_path", None)
    data = source.get("data", {})
    data.pop("root", None)
    normalized = {
        "experiment": experiment,
        "data": data,
        "model": source.get("model", {}),
        "evaluation": source.get("evaluation", {}),
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
