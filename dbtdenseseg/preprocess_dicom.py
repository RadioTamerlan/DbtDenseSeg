"""
DICOM -> NIfTI preprocessing for DbtDenseSeg.

For each DICOM series under --input (whether a single **multi-frame** file or
**many single-frame files, one per slice**) this script:

  1. assembles the slices into one volume (in the model/training orientation),
  2. detects the **view** (CC / MLO / ML ...) and **laterality** (L / R) by
     checking *many* DICOM headers (not just ViewPosition) and finally the
     folder/file name and image content,
  3. writes a single NIfTI whose **file name carries the view**, e.g.
     `…_L_MLO.nii.gz`.

The main pipeline then reads the view straight from that file name (no DICOM
re-parsing). Run it like:

    python preprocess_dicom.py --input <dicom root> --out <nifti root>
    python run_pipeline.py     --input <nifti root> --format both

Why detect from many headers: DBT vendors are inconsistent — Hologic often leaves
ViewPosition / ImageLaterality empty and only encodes the view in
SeriesDescription, a ViewCodeSequence, or the export folder name.
"""
from __future__ import annotations
import argparse
import os
import re
import sys

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import io_volume as V                     # noqa: E402  (no torch import here)
from harmonize_dbt import content_laterality  # noqa: E402
from minerbar import mine                 # noqa: E402

VIEWS = ("XCCL", "XCCM", "MLO", "CC", "ML", "LM", "AT", "CV")   # check longer tokens first
# map ViewCodeSequence / text phrases -> canonical view
_PHRASE = [
    (r"MEDIO.?LATERAL.?OBLIQUE", "MLO"),
    (r"CRANIO.?CAUDAL", "CC"),
    (r"EXAGGERATED.?CRANIO.?CAUDAL", "XCCL"),
    (r"MEDIO.?LATERAL\b", "ML"),
    (r"LATERO.?MEDIAL\b", "LM"),
]
_TEXT_TAGS = ["SeriesDescription", "ProtocolName", "StudyDescription",
              "AcquisitionDeviceProcessingDescription", "ImageComments",
              "PerformedProcedureStepDescription"]


def _view_in(text):
    """Return (view, laterality) parsed from a free-text string, or (None, None)."""
    if not text:
        return None, None
    T = re.sub(r"[^A-Z]", " ", str(text).upper())
    # strongest signal: a laterality letter glued to a view token, e.g. "L MLO", "RCC"
    m = re.search(r"\b([LR])\s?(XCCL|XCCM|MLO|CC|ML|LM)\b", T)
    if m:
        return m.group(2), m.group(1)
    view = None
    for tok in VIEWS:
        if re.search(rf"\b{tok}\b", T):
            view = tok
            break
    if view is None:
        for pat, v in _PHRASE:
            if re.search(pat, T):
                view = v
                break
    lat = None
    if re.search(r"\bLEFT\b", T):
        lat = "L"
    elif re.search(r"\bRIGHT\b", T):
        lat = "R"
    return view, lat


def _from_view_code_seq(ds):
    """View + laterality from ViewCodeSequence / ViewModifierCodeSequence CodeMeaning."""
    view = lat = None
    seq = getattr(ds, "ViewCodeSequence", None)
    if seq:
        for it in seq:
            v, l = _view_in(getattr(it, "CodeMeaning", ""))
            view = view or v
            mod = getattr(it, "ViewModifierCodeSequence", None)
            if mod:
                for m in mod:
                    _, l2 = _view_in(getattr(m, "CodeMeaning", ""))
                    lat = lat or l2
    return view, lat


def detect_view_laterality(files, series_dir, volume=None):
    """Robustly determine (laterality, view, sources) from many DICOM headers,
    then the path, then (laterality only) image content."""
    import pydicom
    dss = []
    for f in files[:8]:
        try:
            dss.append(pydicom.dcmread(f, stop_before_pixels=True, force=True))
        except Exception:
            pass
    view = lat = None
    src = []

    # 1) ViewPosition tag
    for ds in dss:
        vp, _ = _view_in(getattr(ds, "ViewPosition", ""))
        if vp:
            view = vp; src.append("ViewPosition"); break
    # 2) ViewCodeSequence
    for ds in dss:
        v, l = _from_view_code_seq(ds)
        if v and not view:
            view = v; src.append("ViewCodeSequence")
        lat = lat or l
    # 3) ImageLaterality / Laterality tags
    for ds in dss:
        l = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")).strip().upper()
        if l in ("L", "R"):
            lat = lat or l
            if "ImageLaterality" not in src:
                src.append("ImageLaterality")
            break
    # 4) free-text header fields
    for ds in dss:
        for tag in _TEXT_TAGS:
            v, l = _view_in(getattr(ds, tag, ""))
            if v and not view:
                view = v; src.append(tag)
            lat = lat or l
        if view and lat:
            break
    # 5) folder / file names
    if not view or not lat:
        v, l = _view_in(series_dir + " " + " ".join(os.path.basename(f) for f in files))
        if v and not view:
            view = v; src.append("path")
        lat = lat or l
        if l and "path" not in src:
            src.append("path")
    # 6) content fallback for laterality only
    if not lat and volume is not None:
        lat = content_laterality(volume); src.append("content(lat)")
    return lat, view, src


def main():
    ap = argparse.ArgumentParser(description="DICOM -> NIfTI with view-tagged filenames")
    ap.add_argument("--input", required=True, help="root folder of DICOM patient subfolders")
    ap.add_argument("--out", required=True, help="output root for the view-tagged NIfTI files")
    ap.add_argument("--default-view", default="", help="view to assume if none detected (e.g. CC)")
    args = ap.parse_args()

    series = [s for s in V.discover_series(args.input) if s["kind"] == "dcm"]
    print(f"found {len(series)} DICOM series under {args.input}\n", flush=True)
    n_ok = n_noview = 0
    for s in mine(series, desc="preprocessing DICOM"):
        rel = os.path.relpath(s["series_dir"], args.input)
        patient = rel.split(os.sep)[0]
        try:
            vol, meta = V.read_series(s)          # (Z,H,W) in model/training orientation
            lat, view, src = detect_view_laterality(s["payload"], s["series_dir"], vol)
            if not view and args.default_view:
                view = args.default_view.upper(); src.append("default")
            lat = lat or "U"                       # unknown laterality
            if not view:
                view = "UNK"; n_noview += 1
            else:
                n_ok += 1
            nz = vol.shape[0]
            sid = re.sub(r"[^0-9A-Za-z]+", "", s["ident"])[-8:] or "series"
            stem = f"{patient}_{sid}_{nz}_{lat}_{view}"
            outdir = os.path.join(args.out, patient)
            os.makedirs(outdir, exist_ok=True)
            out = os.path.join(outdir, stem + ".nii.gz")
            nib.save(nib.Nifti1Image(np.ascontiguousarray(vol).astype(np.float32),
                                     np.eye(4)), out)
            tile = "n_slices={} files={}".format(nz, len(s["payload"]))
            tqdm_write(f"  {patient}: view={view} lat={lat} via {src or 'NONE'}  {tile}  -> {out}")
        except Exception as e:
            tqdm_write(f"  {patient}: FAILED {e}")
    print(f"\nDone. {n_ok} with view, {n_noview} without (named *_UNK).", flush=True)
    print(f"Now run:  python {os.path.join(HERE, 'run_pipeline.py')} --input {args.out} --format both")


def tqdm_write(msg):
    from tqdm import tqdm
    tqdm.write(msg)


if __name__ == "__main__":
    main()
