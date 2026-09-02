from __future__ import annotations

from torch import nn

from .normalization import make_norm, uses_learned_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, image_channels: int = 1, mask_channels: int = 1, base: int = 64, norm_type: str = "batch"):
        super().__init__()
        in_ch = image_channels + mask_channels

        def block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1, bias=not (norm and uses_learned_norm(norm_type)))]
            if norm:
                layers.append(make_norm(norm_type, oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base, norm=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, image, mask_representation):
        return self.net(__import__("torch").cat([image, mask_representation], dim=1))
