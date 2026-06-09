"""
DBT ensemble inference pipeline
===============================

Input : a root folder of patient subfolders, each containing DICOM series and/or
        NIfTI volumes (one volume per series).
Models: the three winners (area, muscle, dense) + ensemble (dense ∩ area ∩ ¬muscle).
Output: written into  <series folder>/model prediction/  as the ORIGINAL image
        plus the predicted masks, in the format chosen by --format:
          nii  -> NIfTI (.nii.gz)
          dcm  -> Secondary-Capture DICOM (per-slice)
          both -> both

Device: --device auto (GPU if available else CPU). CPU works (slower; AMP off).
Cross-vendor (optional): --harmonize --ref hologic_reference.npz applies intensity
        histogram-matching to the Hologic reference before inference (for GE/Siemens).

Examples:
  python run_pipeline.py --input /data/patients --format both
  python run_pipeline.py --input /data/patients --format nii --device cpu
  python run_pipeline.py --input /data/ge_study --format dcm --harmonize --ref hologic_reference.npz
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback, warnings

# quiet the library noise (HF head-shape notice, monai/pkg_resources deprecation)
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inference as I            # noqa: E402
import io_volume as V           # noqa: E402
from minerbar import mine       # noqa: E402

MASKS = ["ensemble", "dense", "area", "muscle"]


def stem_of(item):
    if item["kind"] == "nii":
        return os.path.basename(item["payload"]).replace(".nii.gz", "").replace(".nii", "")
    return f"series_{item['ident']}"


def write_outputs(item, meta, orig_canon, masks, outroot, fmt):
    """Inputs are in MODEL/canonical orientation. NIfTI output is written in the
    source-nii orientation; DICOM output in the acquisition orientation (portrait)."""
    os.makedirs(outroot, exist_ok=True)
    stem = stem_of(item)
    fa = meta["flip_axes"]
    tmpl = meta.get("source_template")

    # ---- original image ----
    if fmt in ("nii", "both"):
        if item["kind"] == "nii":
            import shutil
            dst = f"{stem}_original.nii.gz" if item["payload"].endswith(".gz") else f"{stem}_original.nii"
            shutil.copy2(item["payload"], os.path.join(outroot, dst))
        else:
            V.write_nifti(V.to_original_orientation(orig_canon, fa), meta["affine"],
                          os.path.join(outroot, f"{stem}_original.nii.gz"))
    if fmt in ("dcm", "both"):
        if item["kind"] == "dcm":
            V.copy_original_dicom(item["payload"], os.path.join(outroot, f"{stem}_original_dicom"))
        else:
            V.write_dicom_series(V.model_to_dcm(orig_canon),
                                 os.path.join(outroot, f"{stem}_original_dicom"),
                                 template_file=tmpl, desc="original", scale=1)

    # ---- predicted masks ----
    for name in MASKS:
        if fmt in ("nii", "both"):
            V.write_nifti(V.to_original_orientation(masks[name], fa).astype(np.uint8),
                          meta["affine"], os.path.join(outroot, f"{stem}_{name}_mask.nii.gz"))
        if fmt in ("dcm", "both"):
            V.write_dicom_series(V.model_to_dcm(masks[name]),
                                 os.path.join(outroot, f"{stem}_{name}_mask_dicom"),
                                 template_file=tmpl, desc=f"{name} mask", scale=255)


def main():
    ap = argparse.ArgumentParser(description="DBT ensemble inference pipeline")
    ap.add_argument("--input", required=True, help="root folder of patient subfolders")
    ap.add_argument("--format", choices=["nii", "dcm", "both"], default="nii")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--harmonize", action="store_true", help="apply intensity harmonization to Hologic ref")
    ap.add_argument("--ref", default=None, help="hologic_reference.npz (required with --harmonize)")
    args = ap.parse_args()

    device = I.pick_device(args.device)
    print(f"device: {device}  (AMP {'on' if device.type=='cuda' else 'off (CPU float32)'})", flush=True)
    if args.harmonize and not args.ref:
        ap.error("--harmonize requires --ref")
    ref = dict(np.load(args.ref)) if args.harmonize else None

    print("loading models (area, muscle, dense)...", flush=True)
    models = I.load_models(device)

    series = V.discover_series(args.input)
    N = len(series)
    print(f"\nidentified {N} series under {args.input}:")
    for it in series:
        print(f"  - {os.path.relpath(it['series_dir'], args.input)}/{stem_of(it)}  [{it['kind']}]")
    print()

    ok = fail = 0
    t0 = time.time()
    for i, item in enumerate(mine(series, desc="mining patients"), 1):
        patient = os.path.relpath(item["series_dir"], args.input).split(os.sep)[0]
        try:
            ts = time.time()
            vol_canon, meta = V.read_series(item)
            view = meta["view"]
            muscle_on = str(view).upper() in ("MLO", "ML")
            steps = (["reorient DICOM->model"] if item["kind"] == "dcm" else []) + ["intensity/1023"]
            if ref is not None:
                steps.append("harmonize->Hologic")
            tqdm.write(f"[{i}/{N}] {patient} / {stem_of(item)}  ({item['kind'].upper()})")
            tqdm.write(f"      dims(Z,H,W)={tuple(vol_canon.shape)}  view={view}  "
                       f"laterality={meta.get('laterality','?')}  vendor={meta.get('vendor','?')}  "
                       f"spacing={meta.get('pixel_spacing')}x{meta.get('slice_spacing')}")
            tqdm.write(f"      preprocess: {', '.join(steps)}; muscle={'ON' if muscle_on else 'OFF (CC)'}  -> analyzing...")
            infer_in = vol_canon
            if ref is not None:
                from harmonize_dbt import normalize_intensity
                infer_in = normalize_intensity(vol_canon, ref)     # intensity only
            masks = I.run_series(infer_in, view, models, device, threshold=args.threshold)
            outroot = os.path.join(item["series_dir"], V.PRED_DIR)
            write_outputs(item, meta, vol_canon, masks, outroot, args.format)
            ok += 1
            tqdm.write(f"      done in {time.time()-ts:.0f}s  vox: dense={int(masks['dense'].sum())} "
                       f"area={int(masks['area'].sum())} muscle={int(masks['muscle'].sum())} "
                       f"ensemble={int(masks['ensemble'].sum())}  -> {outroot}\n")
            # release this series' big arrays before reading the next one
            del masks, vol_canon, infer_in
            gc.collect()
        except Exception as e:
            fail += 1
            tqdm.write(f"[{i}/{N}] {patient} / {stem_of(item)}  FAILED: {e}")
            traceback.print_exc()

    print(f"Done. {ok} ok, {fail} failed, {N} total in {time.time()-t0:.0f}s.", flush=True)


if __name__ == "__main__":
    main()
