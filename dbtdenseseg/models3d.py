"""
3D model factory for dense segmentation.

  swinunetr      : MONAI SwinUNETR, random init.
  swinunetr_ssl  : MONAI SwinUNETR with SSL-pretrained Swin encoder.
                   Weights from Tang et al., self-supervised on 5050 CT volumes.
                   Downloaded once from MONAI's GitHub release and cached.
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn


SSL_URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/"
           "0.8.1/model_swinvit.pt")
SSL_CACHE_DIR = Path.home() / ".cache" / "monai_swinunetr_ssl"


def _ensure_ssl_weights() -> Path:
    SSL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = SSL_CACHE_DIR / "model_swinvit.pt"
    if fp.is_file():
        return fp
    print(f"[SSL] downloading {SSL_URL} -> {fp}...", flush=True)
    import urllib.request
    urllib.request.urlretrieve(SSL_URL, fp.as_posix())
    print(f"[SSL] cached {fp} ({fp.stat().st_size/1e6:.1f} MB)", flush=True)
    return fp


def build_swinunetr(in_channels: int = 1, out_channels: int = 1,
                    img_size=(32, 256, 256), feature_size: int = 48,
                    ssl_pretrain: bool = False,
                    use_checkpoint: bool = True) -> nn.Module:
    from monai.networks.nets import SwinUNETR
    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )
    if ssl_pretrain:
        fp = _ensure_ssl_weights()
        raw = torch.load(fp.as_posix(), map_location="cpu", weights_only=True)
        sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[SSL] loaded SSL pretrain: "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model


def build_model3d(arch: str, in_channels: int = 1, out_channels: int = 1,
                  img_size=(32, 256, 256), feature_size: int = 48,
                  ssl_pretrain: bool = False) -> nn.Module:
    a = arch.lower()
    if a in ("swinunetr", "swinunetr_ssl"):
        return build_swinunetr(in_channels=in_channels,
                               out_channels=out_channels,
                               img_size=tuple(img_size),
                               feature_size=feature_size,
                               ssl_pretrain=ssl_pretrain or a == "swinunetr_ssl")
    raise ValueError(f"Unknown 3D arch: {arch}")
