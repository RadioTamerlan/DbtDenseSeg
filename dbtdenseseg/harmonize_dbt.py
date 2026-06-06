"""
Vendor-aware DBT preprocessing / harmonization
==============================================

The segmentation models were trained on **Hologic** reconstructed DBT volumes.
To run them on **GE** and **Siemens** studies we must map those vendors' volumes
into the same intensity + geometry distribution the models expect, otherwise the
domain shift degrades the predictions.

Pipeline (per series):
  read DICOM (any vendor)
    -> modality LUT (Rescale slope/intercept)
    -> VOI / windowing  (to a common display domain)
    -> photometric fix  (MONOCHROME1 -> invert)
    -> geometry resample (in-plane -> target pixel spacing, slices -> 1 mm)
    -> intensity harmonization to a Hologic reference
         * percentile normalization of the breast foreground, and/or
         * histogram matching to the Hologic reference CDF
    -> canonical volume (Z,H,W), float in [0, PIX_MAX]

IMPORTANT - laterality: we do NOT flip L/R to a canonical side. Our segmentation
models handle both lateralities natively (trained with horizontal-flip
augmentation; the project tested 'standardize_laterality' and found no gain over
hflip aug). That rot90/flip-to-canonical step is a *TomoLIBRA* requirement, not
ours. View (CC vs MLO/ML) is still detected, but only for **muscle gating**
(run the muscle model on MLO/ML, skip CC) -- never to reorient the image.
(Voxel-array orientation is matched to the training convention by the NIfTI
inference path via affine-sign canonicalization; that is distinct from L/R
laterality standardization -- an R breast stays an R breast.)

What this CAN harmonize (controllable): photometric, LUT/processing domain,
voxel geometry, and the global intensity distribution.

What it CANNOT fully remove (physics -> needs augmentation / domain adaptation
/ fine-tuning): detector-noise texture (Hologic/Siemens a-Se direct vs GE CsI
indirect), z-resolution / slice blur (Hologic ~15 deg vs GE ~25 deg vs Siemens
~50 deg scan angle), and vendor reconstruction sharpening.

Vendor quick-reference (from the literature; used only for sane defaults):
  Hologic Selenia Dimensions : a-Se direct, ~0.10-0.14 mm px, 15 deg / 15 proj, 1 mm slices
  GE SenoClaire / Pristina   : CsI indirect, ~0.10 mm px,     ~25 deg,         0.5-1 mm slices
  Siemens Mammomat           : a-Se direct, ~0.085 mm px,     ~50 deg,         1 mm slices

CLI:
  python harmonize_dbt.py build-reference --src <dir of Hologic .nii.gz/dcm> --out hologic_reference.npz
  python harmonize_dbt.py harmonize --input <dicom dir|file|.nii.gz> --ref hologic_reference.npz --out out.nii.gz
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

PIX_MAX = 1023.0                       # model intensity scale (matches training)
TARGET_INPLANE_MM = 0.14               # Hologic tomo recon pixel pitch (resample target)
TARGET_SLICE_MM = 1.0


# --------------------------------------------------------------------------- #
# Vendor detection
# --------------------------------------------------------------------------- #
def detect_vendor(manufacturer: str) -> str:
    m = (manufacturer or "").lower()
    if "hologic" in m or "lorad" in m:
        return "hologic"
    if "ge" in m or "general electric" in m or "senograph" in m:
        return "ge"
    if "siemens" in m:
        return "siemens"
    return "unknown"


# --------------------------------------------------------------------------- #
# DICOM reading (multi-frame OR per-slice series), faithful to the header
# --------------------------------------------------------------------------- #
def read_dicom_series(path: str):
    """Return (vol[Z,H,W] float, meta dict). Handles multiframe + per-slice."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

    def _spacing_from(ds):
        # per-frame functional groups (multiframe) or top-level
        ps = sl = None
        try:
            sh = ds.SharedFunctionalGroupsSequence[0]
            pm = sh.PixelMeasuresSequence[0]
            ps = [float(x) for x in pm.PixelSpacing]
            sl = float(getattr(pm, "SliceThickness", getattr(pm, "SpacingBetweenSlices", 1.0)))
        except Exception:
            if "PixelSpacing" in ds:
                ps = [float(x) for x in ds.PixelSpacing]
            sl = float(getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.0)))
        if ps is None:
            ps = [1.0, 1.0]
        return ps, sl

    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "**", "*.dcm"), recursive=True)))
    if not files:
        raise FileNotFoundError(f"no DICOM under {path}")

    ds0 = pydicom.dcmread(files[0], force=True)
    manuf = str(getattr(ds0, "Manufacturer", ""))
    photometric = str(getattr(ds0, "PhotometricInterpretation", "MONOCHROME2"))
    image_type = list(getattr(ds0, "ImageType", []))
    view = str(getattr(ds0, "ViewPosition", "")).upper()
    hdr_lat = str(getattr(ds0, "ImageLaterality", getattr(ds0, "Laterality", ""))).upper()

    if int(getattr(ds0, "NumberOfFrames", 1)) > 1:        # multi-frame
        raw = ds0.pixel_array.astype(np.float32)          # (Z,H,W)
        raw = apply_modality_lut(raw, ds0)
        try:
            raw = apply_voi_lut(raw, ds0)
        except Exception:
            pass
        ps, sl = _spacing_from(ds0)
        vol = raw
    else:                                                  # per-slice series
        dss = [pydicom.dcmread(f, force=True) for f in files]
        # sort by ImagePositionPatient z if present else InstanceNumber
        def _key(d):
            ipp = getattr(d, "ImagePositionPatient", None)
            return float(ipp[2]) if ipp else float(getattr(d, "InstanceNumber", 0))
        dss.sort(key=_key)
        slices = []
        for d in dss:
            a = apply_modality_lut(d.pixel_array.astype(np.float32), d)
            try:
                a = apply_voi_lut(a, d)
            except Exception:
                pass
            slices.append(a)
        vol = np.stack(slices, 0)
        ps, sl = _spacing_from(dss[0])

    if photometric.upper() == "MONOCHROME1":              # inverted -> flip to MONOCHROME2 sense
        vol = vol.max() - vol

    meta = dict(manufacturer=manuf, vendor=detect_vendor(manuf),
                photometric=photometric, image_type=image_type, view=view,
                header_laterality=hdr_lat, pixel_spacing=ps, slice_spacing=sl)
    return vol.astype(np.float32), meta


# --------------------------------------------------------------------------- #
# Laterality (content-based; header L/R is unreliable in DBT)
#
# NOTE: this is for REPORTING / muscle-gating context only. Our models do NOT
# require laterality standardization (they handle L/R via hflip augmentation),
# so the harmonizer does not flip by default. `standardize_laterality` is kept
# only for TomoLIBRA-style pipelines that require a canonical side.
# --------------------------------------------------------------------------- #
def content_laterality(vol: np.ndarray) -> str:
    """Return 'L' or 'R' = side of the chest wall (where the breast is widest)."""
    mid = vol[vol.shape[0] // 2]
    fg = mid > (mid.max() * 0.1)
    left = fg[:, : fg.shape[1] // 2].sum()
    right = fg[:, fg.shape[1] // 2:].sum()
    return "L" if left >= right else "R"


def standardize_laterality(vol: np.ndarray, want_chestwall: str = "L") -> np.ndarray:
    """OPTIONAL (TomoLIBRA-style): flip columns so the chest wall sits on a fixed
    side. NOT used for our models -- they handle both lateralities natively."""
    return vol if content_laterality(vol) == want_chestwall else vol[:, :, ::-1]


# --------------------------------------------------------------------------- #
# Breast foreground mask (for intensity stats; excludes air/background)
# --------------------------------------------------------------------------- #
def breast_mask(vol: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    thr = np.percentile(vol[vol > 0], 5) if (vol > 0).any() else 0
    m = vol > max(thr, vol.max() * 0.02)
    m = ndimage.binary_fill_holes(m)
    return m


# --------------------------------------------------------------------------- #
# Geometry resampling to a common voxel size
# --------------------------------------------------------------------------- #
def resample(vol, src_zyx, dst_zyx=(TARGET_SLICE_MM, TARGET_INPLANE_MM, TARGET_INPLANE_MM)):
    from scipy import ndimage
    factors = [s / d for s, d in zip(src_zyx, dst_zyx)]
    if all(abs(f - 1) < 1e-3 for f in factors):
        return vol
    return ndimage.zoom(vol, factors, order=1)


# --------------------------------------------------------------------------- #
# Intensity harmonization to a Hologic reference
# --------------------------------------------------------------------------- #
def build_reference(volumes, n_grid=256):
    """Collect breast-foreground intensities from Hologic volumes -> ref CDF."""
    fg_all = []
    for v in volumes:
        m = breast_mask(v)
        fg = v[m]
        if fg.size:
            fg_all.append(np.random.choice(fg, size=min(fg.size, 200_000), replace=False))
    fg_all = np.concatenate(fg_all)
    qs = np.linspace(0, 100, n_grid)
    grid = np.percentile(fg_all, qs)        # intensity at each quantile
    cdf = qs / 100.0
    return dict(grid=grid.astype(np.float32), cdf=cdf.astype(np.float32),
                p_lo=float(np.percentile(fg_all, 1)), p_hi=float(np.percentile(fg_all, 99)))


def normalize_intensity(vol, ref, method="hist"):
    """Map breast foreground to the Hologic reference, scale to [0, PIX_MAX]."""
    m = breast_mask(vol)
    out = np.zeros_like(vol, dtype=np.float32)
    fg = vol[m]
    if fg.size == 0:
        return out
    if method == "percentile":
        lo, hi = np.percentile(fg, 1), np.percentile(fg, 99)
        scaled = np.clip((fg - lo) / max(hi - lo, 1e-6), 0, 1)
        # stretch onto the reference's own [p_lo,p_hi] window then to PIX_MAX
        scaled = scaled * (ref["p_hi"] - ref["p_lo"]) + ref["p_lo"]
        out[m] = scaled
    else:  # histogram matching (rank-preserving map to reference CDF)
        src_q = np.linspace(0, 100, len(ref["grid"]))
        src_grid = np.percentile(fg, src_q)            # source intensity per quantile
        ref_at_q = np.interp(src_q / 100.0, ref["cdf"], ref["grid"])  # ref intensity per quantile
        out[m] = np.interp(fg, src_grid, ref_at_q)
    # final common scale 0..PIX_MAX based on reference window
    out = np.clip(out / max(ref["p_hi"], 1e-6) * PIX_MAX, 0, PIX_MAX)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Top-level harmonize
# --------------------------------------------------------------------------- #
def harmonize_volume(vol, src_zyx, ref, method="hist", force_laterality=None):
    """Harmonize a volume to the Hologic reference.

    force_laterality: leave None for OUR models (no L/R flip -- they handle both
    via augmentation). Set 'L'/'R' only for a TomoLIBRA-style consumer that needs
    a canonical chest-wall side.
    """
    if force_laterality is not None:                      # off by default
        vol = standardize_laterality(vol, force_laterality)
    vol = resample(vol, src_zyx)
    vol = normalize_intensity(vol, ref, method=method)
    return vol


def harmonize_dicom(path, ref, method="hist", force_laterality=None):
    vol, meta = read_dicom_series(path)
    meta["content_laterality"] = content_laterality(vol)   # reported, not applied
    src_zyx = (meta["slice_spacing"], meta["pixel_spacing"][0], meta["pixel_spacing"][1])
    out = harmonize_volume(vol, src_zyx, ref, method=method, force_laterality=force_laterality)
    return out, meta


# --------------------------------------------------------------------------- #
# IO helpers (also accept NIfTI so we can build the ref from our Hologic data)
# --------------------------------------------------------------------------- #
def load_any(path):
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return np.asarray(nib.load(path).dataobj, dtype=np.float32), None
    return read_dicom_series(path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-reference")
    b.add_argument("--src", required=True, help="glob/dir of Hologic .nii.gz (or dicom dirs)")
    b.add_argument("--out", default="hologic_reference.npz")
    b.add_argument("--limit", type=int, default=30)
    h = sub.add_parser("harmonize")
    h.add_argument("--input", required=True)
    h.add_argument("--ref", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--method", choices=["hist", "percentile"], default="hist")
    args = ap.parse_args()

    if args.cmd == "build-reference":
        paths = sorted(glob.glob(args.src))[: args.limit]
        vols = [load_any(p)[0] for p in paths]
        ref = build_reference(vols)
        np.savez(args.out, **ref)
        print(f"reference from {len(vols)} volumes -> {args.out}  "
              f"p_lo={ref['p_lo']:.1f} p_hi={ref['p_hi']:.1f}")
    else:
        ref = dict(np.load(args.ref))
        vol, meta = (harmonize_dicom(args.input, ref, args.method)
                     if not args.input.endswith((".nii", ".nii.gz"))
                     else (harmonize_volume(load_any(args.input)[0],
                                            (TARGET_SLICE_MM, TARGET_INPLANE_MM, TARGET_INPLANE_MM),
                                            ref, args.method), None))
        import nibabel as nib
        nib.save(nib.Nifti1Image(vol, np.diag([1, 1, 1, 1])), args.out)
        if meta:
            run_muscle = str(meta.get("view", "")).upper() in ("MLO", "ML")
            print(f"vendor={meta['vendor']} view={meta['view']} "
                  f"laterality={meta.get('content_laterality','?')} (reported, not flipped) "
                  f"muscle_gate={'ON' if run_muscle else 'OFF (CC)'} "
                  f"spacing={meta['pixel_spacing']}x{meta['slice_spacing']} "
                  f"-> harmonized {vol.shape} -> {args.out}")
        else:
            print(f"harmonized {vol.shape} -> {args.out}")


if __name__ == "__main__":
    main()
