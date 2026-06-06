"""
2D area-segmentation utilities.

Two data sources:
 1. FFDM dataset (737 PNG pairs at 500x500) under ``area/Training`` and
    ``area/TrainingLabels``. Labels are 3-channel PNGs where area = pixels
    with red < 128 (i.e. inverted red channel).
 2. DBT slices: each multi-label patient's series broken into per-slice 2D
    samples. The 3D area mask is loaded with the same affine-alignment fix
    used elsewhere in the codebase (LoadDBTLabelsD's _align_to_image_affine).

Train sources are mixed with **DBT oversampling** so DBT slices get more
weight than their raw count would imply (the target modality at inference
time is DBT, not FFDM).

Validation set is built from the 3 held-out DBT patients (the same val split
used by `multilabel_split` in dbt_seg_lib). Inference is slice-wise: load the
full 3D volume, predict each Z-slice with the 2D model, stack to 3D, and
optionally apply a 1D Gaussian smoothing in Z to clean up inter-slice noise.

Public API:
    build_ffdm_index()            -> list[FFDMSample]
    build_dbt_slice_index(samples)-> list[DBTSliceSample]
    AreaPNGDataset                — 2D dataset over (image, area-mask) pairs
    smooth_z_gaussian(arr, sigma) — 1D Gaussian smooth along axis 0
"""

from __future__ import annotations
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Default location of the FFDM area dataset (PNGs).
#
# The dataset went through a restructure: the original 737-pair layout was
# flat (`area/Training/` and `area/TrainingLabels/`), while the new 2096-pair
# layout (with CC + MLO views, 524 each) is partitioned by density and class:
#     area/Density{1..4}/{Cancer,Negative}/img500/*.png   (images)
#     area/Density{1..4}/{Cancer,Negative}/bw500/*.png    (labels)
# `build_ffdm_index` handles both layouts: if a "Density1" folder exists at
# the root it walks the new layout, otherwise it falls back to the old
# Training / TrainingLabels pair.
FFDM_ROOT_DEFAULT = "/mnt/data/tamerlan/RSNA2026/DBT_segmentation/area"

# Default location of the precomputed DBT slice cache (produced by
# precompute_dbt_slices.py). When this exists we use the .npy slices and
# avoid the very slow `gzip-stream decode entire volume to get one slice`
# pattern at training time.
DBT_SLICE_CACHE_DEFAULT = "/mnt/data/tamerlan/RSNA2026/DBT_segmentation/dbt_slice_cache"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Re-use the existing 3D affine-alignment helper so the area mask of DBT slices
# is loaded in the same coordinate system as the image volume.
from dbt_seg_lib import _align_to_image_affine  # noqa: E402


@dataclass
class FFDMSample:
    image: str   # path to PNG under Training/
    label: str   # path to PNG under TrainingLabels/


@dataclass
class DBTSliceSample:
    patient: str
    series: str
    view: str
    image_path: str   # original.nii.gz
    label_path: str   # area.nii(.gz)
    z: int            # slice index (0..Z-1) of the DBT volume
    # If cache_dir is set, _load_dbt_slice reads pre-extracted .npy files
    # from there instead of decoding the gzipped .nii volume. Roughly 50x
    # faster — produced by precompute_dbt_slices.py.
    cache_dir: Optional[str] = None


# ---------- Index builders ---------------------------------------------- #

def build_ffdm_index(root: str = FFDM_ROOT_DEFAULT) -> list[FFDMSample]:
    """Return image/label pairs from the FFDM area dataset.

    Auto-detects layout: the new Density{1..4}/{Cancer,Negative}/img500
    structure is preferred when present (2096 pairs including MLO); the old
    flat Training / TrainingLabels layout (737 CC-only pairs) is used as
    fallback. `.DS_Store` and any image whose label is missing are skipped.
    """
    root_p = Path(root)
    samples = []
    new_layout = sorted(root_p.glob("Density*/*/img500"))
    if new_layout:
        for img_dir in new_layout:
            # Parallel label dir is the sibling "bw500".
            lbl_dir = img_dir.parent / "bw500"
            if not lbl_dir.is_dir():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                if img_path.name.startswith("."):  # skip .DS_Store etc.
                    continue
                lbl_path = lbl_dir / img_path.name
                if lbl_path.is_file():
                    samples.append(FFDMSample(image=str(img_path), label=str(lbl_path)))
        return samples

    # Fallback to the original flat layout.
    train_dir = root_p / "Training"
    label_dir = root_p / "TrainingLabels"
    if not (train_dir.is_dir() and label_dir.is_dir()):
        raise FileNotFoundError(f"No FFDM data found under {root}")
    for img_path in sorted(train_dir.glob("*.png")):
        if img_path.name.startswith("."):
            continue
        lbl_path = label_dir / img_path.name
        if lbl_path.is_file():
            samples.append(FFDMSample(image=str(img_path), label=str(lbl_path)))
    return samples


def build_dbt_slice_index(dbt_samples,
                          cache_root: str = DBT_SLICE_CACHE_DEFAULT) -> list[DBTSliceSample]:
    """Expand a list of 3D DBT Samples into per-slice 2D samples.

    Only series that have an area annotation contribute.

    If `cache_root` exists for a given series, slice samples carry a
    `cache_dir` pointer so `_load_dbt_slice` can read the small .npy files
    instead of decoding the gzipped .nii volume. Series without a cache
    fall back to the slow path (still works, just slow).
    """
    import nibabel as nib
    cache_root_p = Path(cache_root) if cache_root else None
    out = []
    for s in dbt_samples:
        if not s.area:
            continue
        hdr = nib.load(s.image).header
        z = int(hdr.get_data_shape()[0])
        series_cache = None
        if cache_root_p is not None:
            cand = cache_root_p / f"{s.patient}__{s.series}"
            if cand.is_dir() and (cand / f"img_z{0:03d}.npy").is_file():
                series_cache = str(cand)
        for zi in range(z):
            out.append(DBTSliceSample(
                patient=s.patient, series=s.series, view=s.view,
                image_path=s.image, label_path=s.area, z=zi,
                cache_dir=series_cache,
            ))
    return out


# ---------- Validation metrics ----------------------------------------- #

def compute_seg_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray,
                        muscle_bin: np.ndarray = None) -> dict:
    """Binary segmentation metrics for a 3D pred/gt pair.

    Returns dict with:
        dice       : 2 TP / (2 TP + FP + FN)
        precision  : TP / (TP + FP)              — sensitive to false positives
        recall     : TP / (TP + FN)              — sensitive to misses
        iou        : TP / (TP + FP + FN)         — stricter than Dice
        muscle_fp_rate : (pred & muscle & ¬gt) / muscle, if muscle_bin given
                         — fraction of muscle wrongly predicted as area.
                         NaN when muscle annotation is absent.

    NaN is returned where a denominator is 0 (e.g., empty mask + empty pred).
    """
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())

    def _safe(num, den):
        return float(num / den) if den > 0 else float("nan")

    out = {
        "dice":      _safe(2 * tp, 2 * tp + fp + fn),
        "precision": _safe(tp, tp + fp),
        "recall":    _safe(tp, tp + fn),
        "iou":       _safe(tp, tp + fp + fn),
    }
    if muscle_bin is not None:
        m = muscle_bin.astype(bool)
        fp_in_muscle = int((p & m & ~g).sum())
        muscle_size = int(m.sum())
        out["muscle_fp_rate"] = _safe(fp_in_muscle, muscle_size)
    else:
        out["muscle_fp_rate"] = float("nan")
    return out


# ---------- View filters (Hydra sweep) ---------------------------------- #

def filter_ffdm_by_view(ffdm_samples, view):
    """Keep FFDM samples whose filename matches the requested view.
    view in {"CC", "MLO", None}. None returns the list unchanged.
    Filenames use LCC/RCC/XCC/LMLO/RMLO conventions."""
    if view is None:
        return list(ffdm_samples)
    v = view.upper()
    out = []
    for s in ffdm_samples:
        name = os.path.basename(s.image).upper()
        if v == "CC":
            # CC view: filename contains "CC" but NOT "MLO".
            if "CC" in name and "MLO" not in name:
                out.append(s)
        elif v == "MLO":
            if "MLO" in name:
                out.append(s)
    return out


def filter_dbt_by_view(dbt_samples, view):
    """Keep DBT Samples whose .proj matches the requested view.
    view in {"CC", "MLO", None}."""
    if view is None:
        return list(dbt_samples)
    v = view.upper()
    if v == "CC":
        return [s for s in dbt_samples if s.proj == "CC"]
    if v == "MLO":
        return [s for s in dbt_samples if s.proj in ("MLO", "ML")]
    return list(dbt_samples)


# ---------- Loaders ----------------------------------------------------- #

def apply_clahe(img: np.ndarray, clip_limit: float = 0.03,
                tiles=(8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on a [0,1] float image.

    Returns a float image in [0,1]. Used as a 2nd input channel when
    enhance_contrast=True to highlight subtle muscle-vs-tissue contrast that
    the model otherwise struggles to learn from raw intensity.
    """
    from skimage import exposure
    out = exposure.equalize_adapthist(np.clip(img, 0, 1), clip_limit=clip_limit, nbins=256)
    return out.astype(np.float32)


def _maybe_ffdm_muscle(sample: "FFDMSample", target_size: int = 500):
    """If a muscle PNG exists next to the FFDM image's label (under a sibling
    muscle500/ folder), load and return it as a (H, W) uint8 mask. Else return
    None. Produced by precompute_ffdm_muscle.py.
    """
    lbl_path = Path(sample.label)
    cls_dir = lbl_path.parent.parent   # .../bw500/<name> -> .../Density?/<Cancer|Negative>
    mus_path = cls_dir / "muscle500" / lbl_path.name
    if not mus_path.is_file():
        return None
    from PIL import Image as _Image
    arr = np.asarray(_Image.open(str(mus_path)))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8)
    if arr.shape != (target_size, target_size):
        arr = _resize_slice(arr, target_size, "NEAREST")
    return arr


def _load_ffdm(sample: FFDMSample, target_size: int = 500):
    """Load FFDM PNG image and label. Returns (img_float, mask_uint8) both
    (target_size, target_size). Image normalised to [0,1]."""
    from PIL import Image
    img = np.asarray(Image.open(sample.image))
    if img.ndim == 3:
        img = img[..., 0]  # collapse to grayscale (channels are usually identical)
    img = img.astype(np.float32) / 255.0
    lbl = np.asarray(Image.open(sample.label))
    if lbl.ndim == 3:
        mask = (lbl[..., 0] < 128).astype(np.uint8)
    else:
        mask = (lbl > 0).astype(np.uint8)
    if img.shape != (target_size, target_size):
        from PIL import Image as _I
        img = np.asarray(_I.fromarray((img * 255).astype(np.uint8)).resize(
            (target_size, target_size), _I.BILINEAR), dtype=np.float32) / 255.0
        mask = np.asarray(_I.fromarray(mask).resize(
            (target_size, target_size), _I.NEAREST), dtype=np.uint8)
    return img, mask


def _resize_slice(arr2d: np.ndarray, target_size: int, mode: str = "BILINEAR"):
    """Resize a 2D slice to target_size×target_size."""
    from PIL import Image as _I
    if arr2d.shape == (target_size, target_size):
        return arr2d
    if arr2d.dtype == np.uint8:
        img = _I.fromarray(arr2d)
    else:
        # normalise to uint8 for PIL, then back to float
        a = arr2d.astype(np.float32)
        amax = max(a.max(), 1e-6)
        img = _I.fromarray((np.clip(a / amax, 0, 1) * 255).astype(np.uint8))
    resized = img.resize((target_size, target_size),
                         _I.BILINEAR if mode == "BILINEAR" else _I.NEAREST)
    out = np.asarray(resized)
    if arr2d.dtype != np.uint8:
        out = out.astype(np.float32) / 255.0 * amax
    return out


def _load_dbt_slice(sample: DBTSliceSample, target_size: int = 500,
                    pix_max: float = 1023.0):
    """Load one DBT slice + its (affine-aligned) area mask slice. Returns
    (img_float[H,W], mask_uint8[H,W]) at target_size×target_size.

    Fast path: read pre-extracted .npy slices from sample.cache_dir (produced
    by precompute_dbt_slices.py). ~1 ms per slice.

    Slow path (no cache): decode the full gzipped .nii volume to get one
    slice. Tens of seconds per call. Only used as fallback.
    """
    if sample.cache_dir is not None:
        img_path = os.path.join(sample.cache_dir, f"img_z{sample.z:03d}.npy")
        lbl_path = os.path.join(sample.cache_dir, f"lbl_z{sample.z:03d}.npy")
        img = np.load(img_path).astype(np.float32)
        mask = np.load(lbl_path).astype(np.uint8)
        if img.shape != (target_size, target_size):
            img = _resize_slice(img, target_size, "BILINEAR").astype(np.float32)
            mask = _resize_slice(mask, target_size, "NEAREST")
        img = np.clip(img / pix_max, 0, 1).astype(np.float32)
        return img, mask

    # Slow fallback for series without cache.
    import nibabel as nib
    img_nii = nib.load(sample.image_path)
    img = np.asarray(img_nii.dataobj[sample.z], dtype=np.float32)
    img_aff = np.asarray(img_nii.affine)

    lbl_nii = nib.load(sample.label_path)
    full = np.asarray(lbl_nii.dataobj, dtype=np.float32)
    full = _align_to_image_affine(full, np.asarray(lbl_nii.affine), img_aff)
    mask = (full[sample.z] > 0).astype(np.uint8)

    img = _resize_slice(img, target_size, "BILINEAR")
    img = np.clip(img / pix_max, 0, 1).astype(np.float32)
    mask = _resize_slice(mask, target_size, "NEAREST")
    return img, mask


# ---------- torch.utils.data Dataset ------------------------------------ #

class AreaPNGDataset:
    """PyTorch-compatible dataset that mixes FFDM and DBT-slice samples.

    Each __getitem__ returns a dict with:
        image  : (1, H, W) float32 in [0, 1]
        label  : (1, H, W) float32 binary  (area mask, 1 = breast)
        muscle : (1, H, W) float32 binary  (pectoral muscle mask, 1 = muscle).
                 Zeros for FFDM, DBT CC, or DBT MLO/ML without muscle annotation.
                 Used by muscle-aware loss to penalise area FPs inside muscle.
        source : 0=FFDM, 1=DBT

    `standardize_laterality=True` flips every R-laterality sample horizontally
    on load so the chest wall is always on the LEFT after preprocessing. For
    FFDM it uses the filename's laterality (filenames are reliable, verified
    0% mismatch). For DBT it uses the per-series content-detected laterality
    stored in cache_dir/laterality.txt (DBT filenames had 47% mismatch). With
    this flag on, the random L/R flip augmentation is disabled — the model is
    trained to expect chest wall on the left, period.
    """

    SOURCE_FFDM = 0
    SOURCE_DBT = 1

    def __init__(self, ffdm: list, dbt: list,
                 target_size: int = 500, augment: bool = False,
                 standardize_laterality: bool = False,
                 enhance_contrast: bool = False,
                 clahe_only: bool = False):
        self.ffdm = list(ffdm)
        self.dbt = list(dbt)
        self.target_size = target_size
        self.augment = augment
        self.standardize_laterality = standardize_laterality
        # Three input modes (mutually exclusive):
        #   enhance_contrast: 2-channel (raw, CLAHE), in_channels=2
        #   clahe_only:       1-channel CLAHE, in_channels=1
        #   neither:          1-channel raw,   in_channels=1
        assert not (enhance_contrast and clahe_only), \
            "enhance_contrast and clahe_only are mutually exclusive"
        self.enhance_contrast = enhance_contrast
        self.clahe_only = clahe_only

    def __len__(self):
        return len(self.ffdm) + len(self.dbt)

    def _maybe_muscle(self, sample: "DBTSliceSample"):
        """Load the muscle slice from cache if present, else return zeros."""
        if sample.cache_dir is None:
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        p = os.path.join(sample.cache_dir, f"muscle_z{sample.z:03d}.npy")
        if not os.path.isfile(p):
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        mus = np.load(p).astype(np.uint8)
        if mus.shape != (self.target_size, self.target_size):
            mus = _resize_slice(mus, self.target_size, "NEAREST")
        return mus

    @staticmethod
    def _ffdm_laterality(sample) -> str:
        n = os.path.basename(sample.image)
        if "LCC" in n or "LMLO" in n: return "L"
        if "RCC" in n or "RMLO" in n: return "R"
        return "?"

    @staticmethod
    def _dbt_cached_laterality(ds) -> str:
        """Read content-detected laterality from cache_dir/laterality.txt.
        Falls back to '?' if the file is missing (then no flipping)."""
        if ds.cache_dir is None: return "?"
        p = os.path.join(ds.cache_dir, "laterality.txt")
        if not os.path.isfile(p): return "?"
        try:
            return open(p).read().strip()
        except Exception:
            return "?"

    def __getitem__(self, idx):
        if idx < len(self.ffdm):
            ffdm = self.ffdm[idx]
            img, mask = _load_ffdm(ffdm, self.target_size)
            muscle = _maybe_ffdm_muscle(ffdm, self.target_size)
            if muscle is None:
                muscle = np.zeros_like(mask)
            src = self.SOURCE_FFDM
            sample_lat = self._ffdm_laterality(ffdm)  # filename, reliable
        else:
            ds = self.dbt[idx - len(self.ffdm)]
            img, mask = _load_dbt_slice(ds, self.target_size)
            muscle = self._maybe_muscle(ds)
            src = self.SOURCE_DBT
            # DBT filenames are NOT reliable (47% mismatch w/ image content).
            # Use the content-detected laterality stored in the cache instead.
            sample_lat = self._dbt_cached_laterality(ds)

        # Standardise laterality (flip R → L) if requested.
        if self.standardize_laterality and sample_lat == "R":
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            muscle = np.ascontiguousarray(muscle[:, ::-1])

        if self.augment:
            # Skip random L/R flip when laterality is standardised — the model
            # is supposed to learn a fixed spatial prior in this mode.
            if (not self.standardize_laterality) and random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
                muscle = np.ascontiguousarray(muscle[:, ::-1])
            if random.random() < 0.3:
                img = np.clip(img + np.random.uniform(-0.05, 0.05), 0, 1).astype(np.float32)

        import torch
        if self.enhance_contrast:
            # Stack raw + CLAHE as a 2-channel input. The CLAHE channel
            # exposes subtle muscle/tissue contrast the raw channel lacks.
            clahe = apply_clahe(img)
            image_stack = np.stack([img.astype(np.float32),
                                    clahe.astype(np.float32)], axis=0)  # (2, H, W)
        elif self.clahe_only:
            image_stack = apply_clahe(img)[None].astype(np.float32)      # (1, H, W)
        else:
            image_stack = img[None].astype(np.float32)                   # (1, H, W)
        return {
            "image": torch.from_numpy(image_stack),
            "label": torch.from_numpy(mask[None].astype(np.float32)),
            "muscle": torch.from_numpy(muscle[None].astype(np.float32)),
            "source": src,
        }


def make_dbt_weighted_sampler(ds: AreaPNGDataset, dbt_target_fraction: float = 0.7):
    """Build a WeightedRandomSampler that yields DBT slices with the given
    fraction per epoch. The default 0.7 means ~70% of samples per epoch are
    DBT slices (target modality), 30% are FFDM augmentation.

    Per-sample weight is set so that summing weights over each source
    matches the target fractions. Number of draws per epoch matches
    len(dataset) so step count is stable.
    """
    import torch
    n_ffdm = len(ds.ffdm)
    n_dbt = len(ds.dbt)
    ffdm_w = (1.0 - dbt_target_fraction) / max(n_ffdm, 1)
    dbt_w = dbt_target_fraction / max(n_dbt, 1)
    weights = ([ffdm_w] * n_ffdm) + ([dbt_w] * n_dbt)
    return torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(ds), replacement=True,
    )


# ---------- Loss functions for 2D area training ------------------------ #
#
# All three losses take (logits, target) where both are (B, 1, H, W) float
# tensors. `target` is binary (0 / 1). They return a scalar loss.

def _dice_loss_2d(probs, target, smooth: float = 1e-5):
    """Soft Dice loss reduced to scalar (mean over batch). probs already sigmoid'd."""
    dims = (2, 3)
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2 * inter + smooth) / (denom + smooth)).mean()


def bce_dice_loss(logits, target, muscle=None, smooth: float = 1e-5):
    """Symmetric baseline: per-pixel BCE + soft Dice. Penalises FN and FP equally.

    `muscle` is accepted but ignored — kept in the signature so all 2D loss
    fns share the same call shape (the train loop always passes muscle).
    """
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    return bce + _dice_loss_2d(probs, target, smooth)


def focal_tversky_precision_loss(logits, target, muscle=None,
                                 alpha: float = 0.3,
                                 beta: float = 0.7, gamma: float = 4.0 / 3.0,
                                 smooth: float = 1e-5):
    """Precision-biased Focal Tversky.

    Tversky = TP / (TP + alpha*FN + beta*FP).
    With alpha < beta, a false positive costs more than a false negative —
    the gradient pushes the model toward UNDER-painting. Designed to suppress
    muscle-as-area FPs on MLO views.

    Note: cast the .pow(gamma) into fp32 to avoid AMP fp16 fragility when
    (1 - tversky) is small. This is the same safeguard used in
    dbt_seg_lib.focal_tversky_loss for 3D.
    """
    import torch
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    tp = (probs * target).sum(dim=dims)
    fn = (target * (1.0 - probs)).sum(dim=dims)
    fp = ((1.0 - target) * probs).sum(dim=dims)
    with torch.cuda.amp.autocast(enabled=False):
        tp32, fn32, fp32 = tp.float(), fn.float(), fp.float()
        tversky = (tp32 + smooth) / (tp32 + alpha * fn32 + beta * fp32 + smooth)
        loss = (1.0 - tversky).clamp(min=1e-8).pow(gamma)
    return loss.mean()


def bce_dice_tversky_loss(logits, target, muscle=None,
                          lambda_tversky: float = 0.5,
                          alpha: float = 0.3, beta: float = 0.7,
                          gamma: float = 4.0 / 3.0, smooth: float = 1e-5):
    """Compound: BCE + Dice + lambda_tversky * Focal_Tversky(alpha<beta)."""
    bce_dice = bce_dice_loss(logits, target, smooth=smooth)
    ft = focal_tversky_precision_loss(logits, target, alpha=alpha, beta=beta,
                                      gamma=gamma, smooth=smooth)
    return bce_dice + lambda_tversky * ft


def bce_dice_muscle_aware_loss(logits, target, muscle=None,
                               lambda_muscle: float = 0.3,
                               smooth: float = 1e-5):
    """BCE + Dice + lambda_muscle * (mean prob inside muscle ∩ ¬area_target).

    The muscle mask may come from:
      - DBT muscle.nii (real annotation), or
      - FFDM heuristic detector (precompute_ffdm_muscle.py) — approximate.

    Two safeguards keep the penalty robust against bad muscle masks:

    1. **Consistency filter**: only penalise where `muscle == 1 AND target == 0`.
       If the muscle detector wrongly includes a pixel that the area GT
       actually labels as breast (target=1), we trust the GT and zero the
       penalty there. This is a no-op for the FFDM detector (which already
       requires `not_area`) but adds safety against annotator inconsistencies
       in DBT muscle annotations.

    2. **Lower default lambda (0.3 instead of 1.0)**: with imperfect muscle
       masks, a softer weight prevents the model from being pushed too hard
       away from wrongly-flagged regions. The base BCE+Dice can override the
       muscle signal when the GT clearly disagrees.

    Per-sample penalty:
        muscle_consistent = muscle * (1 - target)
        penalty = mean(prob * muscle_consistent) / (mean(muscle_consistent) + eps)

    For samples without a muscle annotation (e.g., CC), muscle is all-zeros
    and the penalty degenerates to 0 — loss = BCE + Dice.
    """
    import torch
    base = bce_dice_loss(logits, target, smooth=smooth)
    if muscle is None:
        return base
    probs = torch.sigmoid(logits)
    # Consistency filter: zero out muscle pixels that overlap the area GT.
    muscle_cons = muscle * (1.0 - target)
    num = (probs * muscle_cons).sum(dim=(2, 3))
    den = muscle_cons.sum(dim=(2, 3)) + smooth
    penalty = (num / den).mean()
    return base + lambda_muscle * penalty


def bce_dice_tversky_muscle_aware_loss(logits, target, muscle=None,
                                       lambda_tversky: float = 0.5,
                                       alpha: float = 0.3, beta: float = 0.7,
                                       gamma: float = 4.0 / 3.0,
                                       lambda_muscle: float = 0.3,
                                       smooth: float = 1e-5):
    """BCE + Dice + lambda_tversky * Focal_Tversky(α<β) + lambda_muscle * muscle_pen.

    Three precision-pushing signals stacked:
      - BCE + Dice  : anchor for basic area segmentation (stability).
      - Focal Tversky (α=0.3, β=0.7): global precision push — every FP costs
                     more than every FN. Anatomy-blind.
      - Muscle penalty (λ=0.3, consistency-filtered): anatomy-aware extra
                     penalty on area predictions inside muscle ∩ ¬area_target.
    """
    import torch
    bce_dice = bce_dice_loss(logits, target, smooth=smooth)
    ft = focal_tversky_precision_loss(logits, target, alpha=alpha, beta=beta,
                                      gamma=gamma, smooth=smooth)
    base = bce_dice + lambda_tversky * ft
    if muscle is None:
        return base
    probs = torch.sigmoid(logits)
    muscle_cons = muscle * (1.0 - target)
    num = (probs * muscle_cons).sum(dim=(2, 3))
    den = muscle_cons.sum(dim=(2, 3)) + smooth
    penalty = (num / den).mean()
    return base + lambda_muscle * penalty


# Convenience dispatcher used by train_area2d.py
LOSS_CHOICES_2D = ("bce_dice", "bce_dice_tversky",
                   "focal_tversky_precision", "bce_dice_muscle_aware",
                   "bce_dice_tversky_muscle_aware")


def get_loss_fn(name: str):
    if name == "bce_dice":
        return bce_dice_loss
    if name == "bce_dice_tversky":
        return bce_dice_tversky_loss
    if name == "focal_tversky_precision":
        return focal_tversky_precision_loss
    if name == "bce_dice_muscle_aware":
        return bce_dice_muscle_aware_loss
    if name == "bce_dice_tversky_muscle_aware":
        return bce_dice_tversky_muscle_aware_loss
    raise ValueError(f"unknown 2D loss {name!r}; choices: {LOSS_CHOICES_2D}")


# ---------- 1D Gaussian smoothing along Z (post-inference) -------------- #

def smooth_z_gaussian(prob_3d: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply a 1D Gaussian smoothing along axis 0 (Z) of a 3D probability
    volume. `sigma` is in slice units. sigma=0 returns the input unchanged.

    Useful after slice-wise inference to remove flickering inter-slice
    inconsistencies caused by the 2D model treating each slice independently.
    """
    if sigma <= 0:
        return prob_3d
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(prob_3d, sigma=sigma, axis=0, mode="reflect")
