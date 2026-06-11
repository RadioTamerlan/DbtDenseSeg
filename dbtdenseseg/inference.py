"""Model loading + device-aware (GPU/CPU) inference + ensemble.

Reuses the three winner checkpoints (area / muscle / dense) and the shared model
code under code/dense/. Works on CPU when no GPU is present (AMP fp16 is enabled
only on CUDA; CPU runs float32 with a smaller sliding-window batch).
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import numpy as np
import torch
from PIL import Image as PILImage
from minerbar import mine
try:
    import transformers
    transformers.logging.set_verbosity_error()   # hide the SegFormer head-shape notice
except Exception:
    pass

PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG)
from models2d import build_model                       # noqa: E402
from models3d import build_model3d                     # noqa: E402
from area2d_lib import _resize_slice, apply_clahe, smooth_z_gaussian  # noqa: E402

PIX_MAX = 1023.0
# weights live in <repo>/weights by default; override with DBTDENSESEG_WEIGHTS
WEIGHTS_DIR = os.environ.get("DBTDENSESEG_WEIGHTS", os.path.normpath(os.path.join(PKG, "..", "weights")))
AREA_CKPT = os.path.join(WEIGHTS_DIR, "area.pt")
MUSCLE_CKPT = os.path.join(WEIGHTS_DIR, "muscle.pt")
DENSE_CKPT = os.path.join(WEIGHTS_DIR, "dense.pt")


def _check_weights():
    missing = [p for p in (AREA_CKPT, MUSCLE_CKPT, DENSE_CKPT) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model weights:\n  " + "\n  ".join(missing) +
            f"\nPut area.pt / muscle.pt / dense.pt in {WEIGHTS_DIR} "
            "(see weights/README.md) or set DBTDENSESEG_WEIGHTS to their folder.")


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but no CUDA device found")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)


# --------------------------------------------------------------------------- #
def load_segformer(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd.get("cfg", {})
    arch = cfg.get("model", {}).get("arch", "segformer_b2")
    in_ch = int(cfg.get("model", {}).get("in_channels", 1))
    pretrained = cfg.get("model", {}).get("pretrained", None)
    target_size = int(cfg.get("target_size", 512))
    clahe_only = bool(cfg.get("input", {}).get("clahe_only", False))
    model = build_model(arch, in_channels=in_ch, pretrained=pretrained).to(device).eval()
    model.load_state_dict(sd["model"])
    return model, target_size, clahe_only


def load_dense(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = sd.get("args", {})
    roi = tuple(args.get("roi", [32, 256, 256]))
    fs = int(args.get("feature_size", 48))
    model = build_model3d("swinunetr", in_channels=1, out_channels=1,
                          img_size=roi, feature_size=fs, ssl_pretrain=False).to(device).eval()
    model.load_state_dict(sd["model"])
    return model, roi


def load_models(device):
    _check_weights()
    return dict(area=load_segformer(AREA_CKPT, device),
                muscle=load_segformer(MUSCLE_CKPT, device),
                dense=load_dense(DENSE_CKPT, device))


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_2d(model, vol, device, target_size, amp, clahe, desc="2D slices"):
    Z, H, W = vol.shape
    out = np.zeros((Z, H, W), np.float32)
    for z in mine(range(Z), desc=f"mining {desc} slices", leave=False):
        sl = _resize_slice(vol[z], target_size, "BILINEAR")
        sl = np.clip(sl / PIX_MAX, 0, 1).astype(np.float32)
        if clahe:
            sl = apply_clahe(sl).astype(np.float32)
        x = torch.from_numpy(sl[None, None]).to(device)
        with _autocast(device, amp):
            logits = model(x)
        prob = torch.sigmoid(logits[0, 0]).float().cpu().numpy()
        out[z] = np.asarray(PILImage.fromarray((prob * 255).astype(np.uint8)).resize(
            (W, H), PILImage.BILINEAR), np.float32) / 255.0
    return out


@torch.no_grad()
def predict_2d_binary(model, vol, device, target_size, amp, clahe, threshold, desc="2D"):
    """Per-slice 2D prediction thresholded to uint8 **on the fly** — never builds a
    full-resolution float volume, so RAM stays low (one slice at a time)."""
    Z, H, W = vol.shape
    out = np.zeros((Z, H, W), np.uint8)
    for z in mine(range(Z), desc=f"mining {desc} slices", leave=False):
        sl = _resize_slice(vol[z], target_size, "BILINEAR")
        sl = np.clip(sl / PIX_MAX, 0, 1).astype(np.float32)
        if clahe:
            sl = apply_clahe(sl).astype(np.float32)
        x = torch.from_numpy(sl[None, None]).to(device)
        with _autocast(device, amp):
            logits = model(x)
        prob = torch.sigmoid(logits[0, 0]).float().cpu().numpy()
        prob_full = np.asarray(PILImage.fromarray((prob * 255).astype(np.uint8)).resize(
            (W, H), PILImage.BILINEAR), np.float32) / 255.0
        out[z] = (prob_full > threshold).astype(np.uint8)
    return out


@torch.no_grad()
def predict_dense(model, vol, device, roi, amp):
    from monai.inferers import sliding_window_inference
    img = np.clip(vol / PIX_MAX, 0, 1).astype(np.float32)
    x = torch.from_numpy(img[None, None]).to(device)
    sw_batch = 4 if device.type == "cuda" else 1
    with _autocast(device, amp):
        logits = sliding_window_inference(
            inputs=x, roi_size=tuple(roi), sw_batch_size=sw_batch, predictor=model,
            overlap=0.25, mode="gaussian", sw_device=device, device="cpu")
    return torch.sigmoid(logits[0, 0]).float().numpy()


# --------------------------------------------------------------------------- #
def run_series(vol_canon, view, models, device, threshold=0.5, is_3d=True):
    """vol_canon: (Z,H,W). 2D models run per-slice and threshold straight to uint8
    (low RAM). The 3D dense model runs only for 3D inputs (Z>1); a 2D image gets
    area + muscle only (the 3D dense model does not apply)."""
    amp = device.type == "cuda"
    a_m, a_ts, a_cl = models["area"]
    m_m, m_ts, m_cl = models["muscle"]
    d_m, d_roi = models["dense"]

    area_bin = predict_2d_binary(a_m, vol_canon, device, a_ts, amp, a_cl, threshold, "area")
    run_muscle = str(view).upper() in ("MLO", "ML")          # anatomy gating only
    musc_bin = (predict_2d_binary(m_m, vol_canon, device, m_ts, amp, m_cl, threshold, "muscle")
                if run_muscle else np.zeros_like(area_bin))

    if is_3d:
        dprob = predict_dense(d_m, vol_canon, device, d_roi, amp)
        dense_bin = (dprob > threshold).astype(np.uint8); del dprob
        ens_bin = (dense_bin & area_bin & (1 - musc_bin)).astype(np.uint8)
    else:                                                     # 2D image
        dense_bin = np.zeros_like(area_bin)                   # 3D dense N/A
        ens_bin = (area_bin & (1 - musc_bin)).astype(np.uint8)  # breast region
    return dict(area=area_bin, muscle=musc_bin, dense=dense_bin, ensemble=ens_bin,
                muscle_ran=run_muscle, is_3d=is_3d)
