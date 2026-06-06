"""
Model factory for 2D area / muscle segmentation.

After the architecture sweep concluded, only SegFormer-B2 remained as the
winning architecture for both area (val_dice=0.9772) and muscle
(val_dice=0.9580). Other 2D architectures tried (DINOv2+UNet, MedSAM+UNet,
RadImageNet+UNet) were removed during cleanup.

`build_model(arch, in_channels, ...)` returns a model with a uniform forward
interface: input (B, in_channels, H, W) -> logits (B, 1, H, W).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet normalization stats. Applied inside the wrapper so the pretrained
# BatchNorm/LayerNorm running stats receive the input distribution they were
# trained against. Without this, 1-channel->3-channel repeat in [0,1] mismatches
# pretrained BN stats and training can diverge later in cosine LR decay.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)


class SegFormerWrapper(nn.Module):
    """Adapts HuggingFace SegformerForSemanticSegmentation to (B,1,H,W) -> (B,1,H,W).

    - Input: (B, 1, H, W). Repeated to 3 channels for the pretrained patch
      embed (avoids re-initialising the conv stem), then ImageNet-normalized.
    - SegFormer outputs (B, num_labels, H/4, W/4); bilinearly upsampled
      back to (H, W) so downstream code is shape-identical to BasicUNet.
    """

    def __init__(self, hf_model: nn.Module, in_channels: int = 1):
        super().__init__()
        self.model = hf_model
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = _imagenet_norm(x)
        H, W = x.shape[-2:]
        out = self.model(pixel_values=x)
        logits = out.logits  # (B, num_labels=1, H/4, W/4)
        logits = F.interpolate(logits, size=(H, W),
                               mode="bilinear", align_corners=False)
        return logits


def build_segformer(in_channels: int = 1,
                    pretrained: str = "nvidia/segformer-b2-finetuned-ade-512-512"
                    ) -> nn.Module:
    """Build SegFormer-B2 with 1 output class. Loads pretrained weights for
    everything except the final classifier (which is reshaped to num_labels=1)."""
    from transformers import SegformerForSemanticSegmentation
    hf = SegformerForSemanticSegmentation.from_pretrained(
        pretrained,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    return SegFormerWrapper(hf, in_channels=in_channels)


def build_model(arch: str, in_channels: int = 1, **kwargs) -> nn.Module:
    """Dispatch based on architecture name."""
    a = arch.lower()
    if a in ("segformer", "segformer_b2"):
        pretrained = kwargs.get(
            "pretrained", "nvidia/segformer-b2-finetuned-ade-512-512")
        return build_segformer(in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unknown architecture: {arch}")
