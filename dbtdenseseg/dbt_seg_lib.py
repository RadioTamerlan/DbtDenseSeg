"""
Reusable bits for DBT 3D segmentation training (SwinUNETR / MONAI).

Original (single-task) usage stays intact:
    samples = build_index()
    train, val = patient_split(samples, val_frac=0.15, seed=42)
    dicts = to_monai_dicts(train)         # has 'image' + 'label' (= dense_mask/mask.nii.gz)

Multi-task (dense + area + muscle) extensions:
    train, val = combined_split(samples)  # forces 10/3 split for the 13 multi-label patients
    dicts = to_monai_dicts(train)         # also includes 'label_dense', 'label_area', 'label_muscle' paths
    transform = LoadDBTLabelsD()          # builds 3-channel label + 'available' mask

Image source: original.nii.gz under each series in masks/<patient>/<series>/
Dense label : dense_mask/mask.nii.gz under that series
Area  label : area.nii.gz   or   area.nii   (only on patients with multi-label annotation)
Muscle label: muscle.nii.gz or muscle.nii   (only on MLO/ML series of multi-label patients)
"""

from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MASKS_ROOT_DEFAULT = "/mnt/data/tamerlan/RSNA2026/DBT_segmentation/masks"
ORIG_ROOT_DEFAULT = "/mnt/data/tamerlan/RSNA2026/DBT_segmentation/original images"

# Channel ordering used everywhere downstream (model output, label tensor, metrics).
CHANNELS = ("dense", "area", "muscle")


@dataclass
class Sample:
    patient: str
    series: str
    view: str        # "L_CC", "R_MLO", ...
    laterality: str  # "L" or "R"
    proj: str        # "CC", "MLO", "ML", "XCCL", "XCCM"
    image: str       # path to original.nii.gz
    label: str       # path to dense_mask/mask.nii.gz  (kept as the legacy "label")
    dicom_dir: str   # original DICOM dir (may not exist for some)
    area: Optional[str] = None     # path to area.nii(.gz) or None
    muscle: Optional[str] = None   # path to muscle.nii(.gz) or None


def parse_view(series_name: str) -> tuple[str, str, str]:
    """Series_73200000_L_CC_Breast_Tomosynthesis_Image -> ('L_CC', 'L', 'CC')."""
    parts = series_name.split("_")
    laterality = parts[2]
    proj = "_".join(parts[3:-3])
    view = f"{laterality}_{proj}"
    return view, laterality, proj


def _load_bad_files() -> set[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    bf = os.path.join(here, "bad_files.json")
    if not os.path.isfile(bf):
        return set()
    try:
        with open(bf) as f:
            return set(json.load(f).get("paths", []))
    except Exception:
        return set()


def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _align_to_image_affine(arr: np.ndarray,
                           src_affine: np.ndarray,
                           dst_affine: np.ndarray) -> np.ndarray:
    """Flip `arr` along whichever axes have opposite sign in src vs dst affine.

    Our images and dense_mask use affine diag [+1,+1,+1]. area.nii and
    muscle.nii (annotated by a different tool) use [-1,-1,+1] — meaning their
    voxel arrays are mirrored along axes 0 and 1 in world space. Without
    correction, label[i,j,k] does not correspond to image[i,j,k]; the model
    trains on a mirror-image task.

    This helper compares the SIGN of the diagonal of each affine and applies
    np.flip on axes where they differ. For all data we've inspected the affines
    are pure diagonal (no rotation), so sign comparison is sufficient.
    """
    src_signs = np.sign(np.diag(src_affine)[:3])
    dst_signs = np.sign(np.diag(dst_affine)[:3])
    for axis, (s, d) in enumerate(zip(src_signs, dst_signs)):
        if s != 0 and d != 0 and s != d:
            arr = np.flip(arr, axis=axis)
    return np.ascontiguousarray(arr)


def build_index(
    masks_root: str = MASKS_ROOT_DEFAULT,
    orig_root: str = ORIG_ROOT_DEFAULT,
    require_dicom: bool = False,
    keep_views: Optional[set[str]] = None,
) -> list[Sample]:
    """Walk the masks/ tree and return a list of Sample records.

    Series whose image or dense-label path appears in code/bad_files.json (e.g.
    truncated gzip) are skipped automatically. Area / muscle paths are populated
    when present (both .nii.gz and .nii are accepted).
    """
    bad = _load_bad_files()
    samples: list[Sample] = []
    for pat in sorted(os.listdir(masks_root)):
        pdir = os.path.join(masks_root, pat)
        if not os.path.isdir(pdir):
            continue
        for series in sorted(os.listdir(pdir)):
            sdir = os.path.join(pdir, series)
            if not os.path.isdir(sdir):
                continue
            view, lat, proj = parse_view(series)
            if keep_views is not None and view not in keep_views:
                continue
            img = os.path.join(sdir, "original.nii.gz")
            lbl = os.path.join(sdir, "dense_mask", "mask.nii.gz")
            if not (os.path.isfile(img) and os.path.isfile(lbl)):
                continue
            if img in bad or lbl in bad:
                continue
            ddir = os.path.join(orig_root, pat, series)
            if require_dicom and not os.path.isdir(ddir):
                continue
            area_p = _first_existing(
                os.path.join(sdir, "area.nii.gz"),
                os.path.join(sdir, "area.nii"),
            )
            muscle_p = _first_existing(
                os.path.join(sdir, "muscle.nii.gz"),
                os.path.join(sdir, "muscle.nii"),
            )
            samples.append(Sample(
                patient=pat, series=series, view=view,
                laterality=lat, proj=proj,
                image=img, label=lbl, dicom_dir=ddir,
                area=area_p, muscle=muscle_p,
            ))
    return samples


def patient_split(
    samples: list[Sample],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Random split at patient level (no patient appears in both)."""
    pats = sorted({s.patient for s in samples})
    rng = random.Random(seed)
    rng.shuffle(pats)
    n_val = max(1, int(round(val_frac * len(pats))))
    val_pats = set(pats[:n_val])
    train = [s for s in samples if s.patient not in val_pats]
    val = [s for s in samples if s.patient in val_pats]
    return train, val


def multilabel_split(
    samples: list[Sample],
    val_n: int = 3,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Patient-level split over the multi-label patients ONLY.

    Used for single-class training of `area` / `muscle` heads, which only
    have ground truth on the 13 patients carrying area annotations. Patients
    are sorted deterministically, seed-shuffled, and the first `val_n` go to
    val. With the current data this yields 10 train / 3 val.

    The returned series include EVERY view of those 13 patients (CC + MLO +
    XCC*). For the muscle model, `LoadDBTLabelsD` treats CC/XCC series as
    known-empty negatives — anatomically muscle is never visible on CC.
    """
    multi_pats = sorted({s.patient for s in samples if s.area})
    rng = random.Random(seed)
    rng.shuffle(multi_pats)
    val_pats = set(multi_pats[:val_n])
    train_pats = set(multi_pats) - val_pats
    train = [s for s in samples if s.patient in train_pats]
    val = [s for s in samples if s.patient in val_pats]
    return train, val


def combined_split(
    samples: list[Sample],
    val_frac: float = 0.15,
    seed: int = 42,
    multilabel_val_n: int = 3,
) -> tuple[list[Sample], list[Sample]]:
    """Patient-level split that handles dense-only and multi-label patients separately.

    - A patient is "multi-label" if any of their series carries an `area` annotation.
    - Multi-label patients: deterministic seeded shuffle, first `multilabel_val_n` -> val,
      remainder -> train. With the current data this gives 10 train / 3 val.
    - Dense-only patients: seeded shuffle, `val_frac` to val (default 15%).
    - Returns combined train / val sample lists. The dense head learns from every series
      in train; the area/muscle heads learn only from series whose annotation is present
      (handled downstream by the per-channel availability mask in LoadDBTLabelsD).
    """
    by_pat: dict[str, list[Sample]] = {}
    for s in samples:
        by_pat.setdefault(s.patient, []).append(s)

    multi_pats = sorted(p for p, ss in by_pat.items() if any(s.area for s in ss))
    rest_pats = sorted(p for p in by_pat if p not in set(multi_pats))

    rng = random.Random(seed)
    rng.shuffle(multi_pats)
    multi_val = set(multi_pats[:multilabel_val_n])

    rng2 = random.Random(seed)
    rng2.shuffle(rest_pats)
    n_rest_val = max(1, int(round(val_frac * len(rest_pats))))
    rest_val = set(rest_pats[:n_rest_val])

    val_pats = multi_val | rest_val
    train = [s for s in samples if s.patient not in val_pats]
    val = [s for s in samples if s.patient in val_pats]
    return train, val


def to_monai_dicts(samples: list[Sample]) -> list[dict]:
    """Convert Sample list to MONAI's dict-of-paths format.

    Includes both the legacy `label` key (= dense mask, for backward compat with
    smoke_test.py / inference.py / run_val_inference.py) and the new per-channel
    keys consumed by `LoadDBTLabelsD`. Missing labels are stored as empty strings
    so the dict shape stays uniform across batches.
    """
    return [
        {
            "image": s.image,
            "label": s.label,                 # legacy single-channel dense
            "label_dense": s.label,
            "label_area": s.area or "",
            "label_muscle": s.muscle or "",
            "view": s.view,
            "laterality": s.laterality,
            "patient": s.patient,
            "series": s.series,
        }
        for s in samples
    ]


# ---------- Multi-channel loader transform ------------------------------- #

class LoadDBTLabelsD:
    """MONAI-compatible map transform.

    Replaces LoadImaged + EnsureChannelFirstd + the per-channel concat:
    - Loads the image as (1, Z, H, W) float32.
    - Loads `label_dense`, `label_area`, `label_muscle` and stacks them as a
      (3, Z, H, W) float32 binary tensor. Missing labels become all-zero channels.
    - Stores `available` as a (3,) float32 vector with 1.0 where the channel
      had a real annotation and 0.0 where we filled zeros.
    - Channel order matches `CHANNELS = ("dense", "area", "muscle")`.

    All other dict keys are passed through unchanged (so `patient`, `view`, etc.
    survive cropping and end up in the validation loop).
    """

    def __init__(self, channels=CHANNELS, aux_priors=()):
        # The MONAI LoadImage is created lazily on first __call__ so each
        # forked DataLoader worker initialises its own loader state.
        # Sharing the parent's instance across fork has caused intermittent
        # hangs / memory issues with multi-volume label loads.
        # `channels` selects which heads to load — pass `("dense",)` for
        # single-class (dense-only) training.
        # `aux_priors` names channels (e.g. ("muscle",)) loaded as SEPARATE
        # `<name>_prior` keys (and `<name>_prior_avail` scalars) rather than as
        # output channels. Used for anatomy-aware loss penalties on a
        # single-output model (e.g. dense + muscle-FP penalty) without turning
        # the model multi-output. They ride through cropping/flipping like any
        # spatial key, so they stay aligned with the cropped image/label patch.
        self._load = None
        self.channels = tuple(channels)
        self.aux_priors = tuple(aux_priors)

    def __call__(self, data: dict) -> dict:
        # Use nibabel directly so we can access each file's affine and detect
        # orientation mismatches between the image and its labels. MONAI's
        # LoadImage returns a MetaTensor, but extracting the affine and applying
        # custom flips through it is fragile across versions — nibabel is
        # simpler and reliable.
        import nibabel as nib
        d = dict(data)
        img_nii = nib.load(d["image"])
        img_arr = np.asarray(img_nii.dataobj, dtype=np.float32)
        img_aff = np.asarray(img_nii.affine, dtype=np.float64)
        img_arr = img_arr[None]  # add channel dim -> (1, Z, H, W)
        ref_shape = img_arr.shape[1:]

        # Anatomical prior: pectoral muscle is visible only in MLO/ML views.
        # CC and XCC* views have NO muscle by anatomy, so when training a
        # muscle model we treat their (missing) muscle annotation as a known
        # empty mask + available=1, giving the model explicit negative
        # supervision. Without this, CC series for the same multi-label patient
        # would be discarded (available=0) and the model would never learn
        # "predict zero on CC".
        view = d.get("view", "")
        proj = view.split("_", 1)[1] if "_" in view else view
        muscle_anatomically_present = proj in ("MLO", "ML")

        chans = []
        avail = []
        for ch in self.channels:
            path = d.get(f"label_{ch}", "") or ""
            if path and os.path.isfile(path):
                lbl_nii = nib.load(path)
                arr = np.asarray(lbl_nii.dataobj, dtype=np.float32)
                # Align label array to the image's voxel ordering. area.nii.gz
                # and muscle.nii.gz are stored with affine [-1,-1,+1] while the
                # image is [+1,+1,+1] — the array is mirrored along axes 0 and
                # 1 in world space. Without this, the model trains on label
                # data that's misaligned with the image (verified: previous
                # area training plateaued at Dice ~0.45 with predictions that
                # looked mirrored relative to GT).
                arr = _align_to_image_affine(arr, np.asarray(lbl_nii.affine), img_aff)
                if arr.shape != ref_shape:
                    raise ValueError(
                        f"shape mismatch for label_{ch}: got {arr.shape}, "
                        f"image is {ref_shape} (path={path})"
                    )
                arr = (arr > 0).astype(np.float32)
                avail.append(1.0)
            elif ch == "muscle" and not muscle_anatomically_present:
                # CC / XCC* series: muscle anatomically impossible -> known empty.
                arr = np.zeros(ref_shape, dtype=np.float32)
                avail.append(1.0)
            else:
                arr = np.zeros(ref_shape, dtype=np.float32)
                avail.append(0.0)
            chans.append(arr)

        label = np.stack(chans, axis=0).astype(np.float32)        # (3, Z, H, W)
        d["image"] = img_arr
        d["label"] = label
        d["available"] = np.asarray(avail, dtype=np.float32)      # (3,)

        # Auxiliary priors: load named masks as separate (1, Z, H, W) keys for
        # anatomy-aware loss penalties (not as model output channels).
        for ch in self.aux_priors:
            path = d.get(f"label_{ch}", "") or ""
            if path and os.path.isfile(path):
                pn = nib.load(path)
                arr = _align_to_image_affine(
                    np.asarray(pn.dataobj, dtype=np.float32),
                    np.asarray(pn.affine), img_aff)
                if arr.shape != ref_shape:
                    raise ValueError(
                        f"shape mismatch for prior label_{ch}: got {arr.shape}, "
                        f"image is {ref_shape} (path={path})")
                arr = (arr > 0).astype(np.float32)
                av = 1.0
            elif ch == "muscle" and not muscle_anatomically_present:
                # CC/XCC*: muscle anatomically absent -> known-empty prior.
                arr = np.zeros(ref_shape, dtype=np.float32)
                av = 1.0
            else:
                arr = np.zeros(ref_shape, dtype=np.float32)
                av = 0.0
            d[f"{ch}_prior"] = arr[None]                           # (1, Z, H, W)
            d[f"{ch}_prior_avail"] = np.asarray(av, dtype=np.float32)
        # crop_label: single-channel reference for RandCropByPosNegLabeld.
        # We can't pass the 3-channel label directly — MONAI assumes multi-
        # channel labels are one-hot and strips channel 0 as background, which
        # would silently make every dense-only series look empty.
        # Channel 0 (dense) is always present (filtered in build_index) and
        # always has foreground, so using a *view* of it is sufficient: every
        # volume has valid crop centres and we avoid the cost (~1.5 GB) of
        # materialising a union mask in each worker. Positive crops are biased
        # toward dense tissue; area covers dense by construction, and muscle
        # gets enough exposure via random crops + L/R flip augmentation.
        d["crop_label"] = label[:1]                                # (1, Z, H, W) view
        return d


class ComputeSDTd:
    """Compute per-channel signed distance transform of the cropped label.

    Required by the boundary-loss strategy (Kervadec et al. 2019). SDT is
    positive outside the foreground, negative inside, normalized to roughly
    [-1, 1] per crop so the boundary term has a sane magnitude versus the
    region term. Channels with no foreground (or no background) get a zero
    SDT so they don't push gradients in either direction.

    Run AFTER cropping so the distance transform is cheap (small volume).
    """

    def __init__(self, label_key: str = "label", out_key: str = "label_sdt",
                 normalize: bool = True):
        self.label_key = label_key
        self.out_key = out_key
        self.normalize = normalize

    def __call__(self, data: dict) -> dict:
        from scipy.ndimage import distance_transform_edt
        d = dict(data)
        lbl = np.asarray(d[self.label_key])
        # Squeeze any leading singleton dims that MONAI sometimes adds during cropping.
        sdt = np.zeros_like(lbl, dtype=np.float32)
        for c in range(lbl.shape[0]):
            m = lbl[c] > 0.5
            if m.any() and (~m).any():
                pos = distance_transform_edt(~m)
                neg = -distance_transform_edt(m)
                sd = (pos + neg).astype(np.float32)
                if self.normalize:
                    denom = max(1.0, float(np.abs(sd).max()))
                    sd = sd / denom
                sdt[c] = sd
            # else: leave zeros
        d[self.out_key] = sdt
        return d


# ---------- Multi-task loss functions ------------------------------------ #
#
# All losses operate on:
#   logits     : (B, 3, Z, H, W)  -- raw model outputs
#   target     : (B, 3, Z, H, W)  -- binary ground truth
#   available  : (B, 3)           -- 1.0 where a real annotation exists
#
# Per-channel terms are computed first, then masked by `available` and averaged
# only over present (sample, channel) pairs. This means a dense-only series
# contributes ONLY to the dense head's gradient; the area/muscle heads see no
# spurious "all-zero" targets. With ~199 dense-annotated patients and 10 multi-
# label patients, the dense head still gets the bulk of the optimizer's attention.

def _per_channel_dice_loss(probs, target, smooth: float = 1e-5):
    import torch
    dims = (2, 3, 4)
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def _per_channel_bce(logits, target):
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(2, 3, 4))


def dicece_loss(logits, target, available, smooth: float = 1e-5, lambda_ce: float = 1.0):
    """Plain Dice + CE per channel, with availability masking.

    The simplest possible multi-task loss — useful as a reference point against
    which size-rebalancing (gdl_ce), difficulty-rebalancing (focal_tversky), or
    boundary supervision (dicece_boundary) can be judged. If a fancier loss
    isn't beating this, the fancier loss isn't earning its complexity.
    """
    import torch
    probs = torch.sigmoid(logits)
    dice = _per_channel_dice_loss(probs, target, smooth)   # (B, C)
    bce = _per_channel_bce(logits, target)                 # (B, C)
    per_chan = dice + lambda_ce * bce
    mask = available.float()
    n = mask.sum().clamp(min=1.0)
    return (per_chan * mask).sum() / n


def gdl_ce_loss(logits, target, available, smooth: float = 1e-5, lambda_ce: float = 0.5):
    """Generalized Dice (Sudre 2017) + per-channel BCE, with availability masking.

    Reweights each class by 1 / (V_c + 1)^2 where V_c is the foreground voxel
    count of channel c in this batch. Small classes (muscle) and medium classes
    (dense) carry more weight than the large area class. Combined with BCE for
    voxel-level confidence calibration.
    """
    import torch
    probs = torch.sigmoid(logits)
    dims = (2, 3, 4)
    target_sum = target.sum(dim=dims)                      # (B, C)
    inter = (probs * target).sum(dim=dims)                 # (B, C)
    union = probs.sum(dim=dims) + target.sum(dim=dims)     # (B, C)

    w = 1.0 / (target_sum + 1.0).pow(2)                    # (B, C)
    mask = available.float()                               # (B, C)

    num = (w * inter * mask).sum(dim=1)                    # (B,)
    den = (w * union * mask).sum(dim=1)                    # (B,)
    gdl_per_sample = 1.0 - (2.0 * num + smooth) / (den + smooth)

    bce_per_chan = _per_channel_bce(logits, target)        # (B, C)
    n_chan = mask.sum(dim=1).clamp(min=1.0)                # (B,)
    bce_per_sample = (bce_per_chan * mask).sum(dim=1) / n_chan

    sample_mask = (mask.sum(dim=1) > 0).float()
    n_samp = sample_mask.sum().clamp(min=1.0)
    return ((gdl_per_sample + lambda_ce * bce_per_sample) * sample_mask).sum() / n_samp


def focal_tversky_loss(logits, target, available, alpha: float = 0.7,
                       beta: float = 0.3, gamma: float = 4.0 / 3.0,
                       smooth: float = 1e-5):
    """Focal Tversky (Abraham & Khan 2019).

    Tversky_c = TP / (TP + alpha*FN + beta*FP); penalises FN harder than FP
    (alpha > beta). Focal exponent gamma>1 pushes more gradient toward channels
    that are still far from converged.
    """
    import torch
    probs = torch.sigmoid(logits)
    dims = (2, 3, 4)
    # The fractional exponent in `(1 - tversky).pow(gamma)` is the only
    # fp16-fragile operation among the four losses. Under AMP autocast,
    # tiny bases (near-zero or near-one) can produce subnormal numbers and
    # noisy gradients that the optimizer handles poorly. Cast to fp32 just
    # for this region — cost is trivial, but it eliminates the only
    # numerical anomaly that's specific to this loss.
    with torch.cuda.amp.autocast(enabled=False):
        tp = (probs.float() * target.float()).sum(dim=dims)
        fn = (target.float() * (1.0 - probs.float())).sum(dim=dims)
        fp = ((1.0 - target.float()) * probs.float()).sum(dim=dims)
        tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
        # Clamp the base to a strictly positive value so .pow with a
        # fractional exponent can't see a slightly-negative numerical fluke.
        loss = (1.0 - tversky).clamp(min=1e-8).pow(gamma)  # (B, C)

    mask = available.float()
    n = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / n


def dicece_boundary_loss(logits, target, available, target_sdt,
                         lambda_b: float = 0.3, smooth: float = 1e-5):
    """DiceCE + Boundary loss (Kervadec et al. 2019).

    target_sdt: (B, 3, Z, H, W) signed distance map (positive outside).
    Boundary term integrates probability against the SDT, so probability mass
    placed away from the GT boundary directly increases the loss. Aligned with
    NSD / HD95 evaluation.
    """
    import torch
    probs = torch.sigmoid(logits)
    dice = _per_channel_dice_loss(probs, target, smooth)   # (B, C)
    bce = _per_channel_bce(logits, target)                 # (B, C)
    bdry = (probs * target_sdt).mean(dim=(2, 3, 4))        # (B, C)

    per_chan = dice + bce + lambda_b * bdry
    mask = available.float()
    n = mask.sum().clamp(min=1.0)
    return (per_chan * mask).sum() / n


def combo_loss(logits, target, available, target_sdt,
               weights=(0.25, 0.25, 0.25, 0.25)):
    """Weighted sum of the four base strategies.

    weights: (w_dicece, w_gdl_ce, w_focal_tversky, w_dicece_boundary).
    Default equal mix (0.25 each). Each base loss is bounded but on a
    different scale; equal weights treat them as a "vote of confidence"
    rather than a calibrated combination. Pass other weights to bias.
    """
    w_dc, w_gdl, w_ft, w_b = weights
    L_dc  = dicece_loss(logits, target, available)
    L_gdl = gdl_ce_loss(logits, target, available)
    L_ft  = focal_tversky_loss(logits, target, available)
    L_b   = dicece_boundary_loss(logits, target, available, target_sdt)
    return w_dc * L_dc + w_gdl * L_gdl + w_ft * L_ft + w_b * L_b




# ---------- DICOM-side helper for inference (unchanged from original) ---- #
_CANDIDATES = {
    "transpose":           lambda a: np.transpose(a, (0, 2, 1)),
    "transpose_flip_h":    lambda a: np.transpose(a, (0, 2, 1))[:, :, ::-1],
    "transpose_flip_v":    lambda a: np.transpose(a, (0, 2, 1))[:, ::-1, :],
    "transpose_flip_v_h":  lambda a: np.transpose(a, (0, 2, 1))[:, ::-1, ::-1],
}


def detect_dicom_transform(dicom_pixels: np.ndarray, reference: np.ndarray,
                           tol: float = 1e-3) -> Optional[str]:
    ref = reference.astype(np.float32)
    pix = dicom_pixels.astype(np.float32)
    for name, fn in _CANDIDATES.items():
        cand = fn(pix)
        if cand.shape != ref.shape:
            continue
        if np.abs(cand - ref).mean() < tol:
            return name
    return None


def apply_dicom_transform(dicom_pixels: np.ndarray, name: str) -> np.ndarray:
    if name not in _CANDIDATES:
        raise ValueError(f"unknown transform '{name}'; choices: {list(_CANDIDATES)}")
    return np.ascontiguousarray(_CANDIDATES[name](dicom_pixels))


def dicom_to_aligned_volume(
    dicom_path: str,
    transform: Optional[str] = None,
    reference_nii: Optional[str] = None,
) -> np.ndarray:
    """Load a multi-frame DBT DICOM and return it in the model's input frame."""
    import pydicom

    ds = pydicom.dcmread(dicom_path)
    pix = ds.pixel_array  # (Z, H, W)

    if transform is None and reference_nii is not None:
        import nibabel as nib
        ref = nib.load(reference_nii).get_fdata()
        transform = detect_dicom_transform(pix, ref)
        if transform is None:
            raise RuntimeError(
                f"Could not align DICOM {dicom_path} to reference {reference_nii}; "
                f"none of the 4 candidate transforms matched.")
    if transform is None:
        transform = "transpose_flip_h"

    return apply_dicom_transform(pix, transform)
