from __future__ import annotations

from torch import nn


def make_norm(norm_type: str, channels: int) -> nn.Module:
    """Create a 2-D normalization layer with batch-size-one-safe options."""
    kind = str(norm_type).lower()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
    if kind == "group":
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if kind in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unknown normalization {norm_type!r}; choose batch, instance, group, or none")


def uses_learned_norm(norm_type: str) -> bool:
    return str(norm_type).lower() not in {"none", "identity"}
