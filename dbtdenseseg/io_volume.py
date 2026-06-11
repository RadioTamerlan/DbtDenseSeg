"""I/O for the inference pipeline: discover series, read volumes (DICOM or NIfTI),
canonicalize orientation, and write outputs (NIfTI and/or Secondary-Capture DICOM).

Note: DICOM mask output is written as **Secondary Capture** (pydicom), not a
formal DICOM-SEG, because `highdicom` is not installed in this env. It preserves
patient/study/geometry from the source when available and overlays in viewers.
"""
from __future__ import annotations
import os, sys, glob, shutil, re
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonize_dbt import read_dicom_series, content_laterality   # noqa: E402

TRAIN_AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])
_NII_RE = re.compile(r".*\.nii(\.gz)?$", re.I)
PRED_DIR = "model prediction"


# --------------------------------------------------------------------------- #
# Discovery: root -> patient subfolders -> series (nii files or DICOM series)
# --------------------------------------------------------------------------- #
def discover_series(root):
    """Yield dicts: {kind, ident, series_dir, payload}. Skips existing outputs."""
    out = []
    # NIfTI series (each file is a series)
    for p in sorted(glob.glob(os.path.join(root, "**", "*.nii*"), recursive=True)):
        if PRED_DIR in p:
            continue
        out.append(dict(kind="nii", ident=os.path.basename(p), series_dir=os.path.dirname(p), payload=p))
    # DICOM series (group .dcm by SeriesInstanceUID within each dir)
    import pydicom
    dcm_dirs = sorted(set(os.path.dirname(p) for p in
                          glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True)
                          if PRED_DIR not in p))
    for d in dcm_dirs:
        groups = {}
        for f in sorted(glob.glob(os.path.join(d, "*.dcm"))):
            try:
                uid = str(pydicom.dcmread(f, stop_before_pixels=True, force=True).SeriesInstanceUID)
            except Exception:
                uid = d
            groups.setdefault(uid, []).append(f)
        for uid, files in groups.items():
            out.append(dict(kind="dcm", ident=uid[-12:], series_dir=d, payload=files))
    return out


# --------------------------------------------------------------------------- #
# Reading + canonicalization
# --------------------------------------------------------------------------- #
def _view_from_name(name):
    m = re.search(r"_(CC|MLO|ML)\b", name, re.I)
    return m.group(1).upper() if m else None


def read_series(item):
    """Return (vol[Z,H,W] float32, meta). meta carries affine, view, flip_axes, vendor, source."""
    if item["kind"] == "nii":
        nii = nib.load(item["payload"])
        vol = to_zhw_gray(np.asarray(nii.dataobj, dtype=np.float32))
        aff = np.asarray(nii.affine, dtype=np.float64)
        flip_axes = [ax for ax in range(3)
                     if np.sign(np.diag(aff)[ax]) not in (0,)
                     and np.sign(np.diag(aff)[ax]) != np.sign(np.diag(TRAIN_AFFINE)[ax])]
        vol_canon = vol.copy()
        for ax in flip_axes:
            vol_canon = np.flip(vol_canon, axis=ax)
        vol_canon = np.ascontiguousarray(vol_canon)
        meta = dict(kind="nii", affine=aff, flip_axes=flip_axes,
                    view=_view_from_name(item["ident"]), vendor="unknown",
                    is_3d=vol_canon.shape[0] > 1,
                    laterality=content_laterality(vol_canon),
                    pixel_spacing=[round(float(x), 3) for x in nii.header.get_zooms()[:2]],
                    slice_spacing=round(float(nii.header.get_zooms()[2]), 3) if nii.ndim >= 3 else 1.0,
                    source=item["payload"])
        return vol_canon, meta
    else:  # dicom
        vol, m = read_dicom_series(item["payload"][0] if len(item["payload"]) == 1
                                   else os.path.dirname(item["payload"][0]))
        vol = dcm_to_model(to_zhw_gray(vol)).astype(np.float32)   # -> model/training orientation
        try:    # robust multi-header view/laterality (ViewCodeSequence, SeriesDescription, ...)
            from preprocess_dicom import detect_view_laterality
            lat2, view2, _ = detect_view_laterality(item["payload"], item["series_dir"], vol)
        except Exception:
            lat2 = view2 = None
        meta = dict(kind="dcm", affine=np.diag([1.0, 1.0, 1.0, 1.0]), flip_axes=[],
                    view=(m.get("view") or view2 or _view_from_name(item["series_dir"])),
                    vendor=(m.get("vendor") or "unknown"),
                    is_3d=vol.shape[0] > 1,
                    laterality=(m.get("header_laterality") or lat2 or content_laterality(vol)),
                    pixel_spacing=m.get("pixel_spacing", [1.0, 1.0]),
                    slice_spacing=m.get("slice_spacing", 1.0),
                    source=item["payload"], source_template=item["payload"][0])
        return np.ascontiguousarray(vol), meta


def to_original_orientation(arr_canon, flip_axes):
    a = arr_canon
    for ax in flip_axes:
        a = np.flip(a, axis=ax)
    return np.ascontiguousarray(a)


# DICOM acquisition orientation <-> model/training (nii) orientation.
# Verified by exact pixel match (mean|diff|=0): training nii = swapaxes(dcm,1,2)[:, ::-1, :].
def dcm_to_model(v):
    """DICOM (Z, rows, cols) -> model/training orientation (matches the nii)."""
    return np.ascontiguousarray(np.swapaxes(v, 1, 2)[:, ::-1, :])


def model_to_dcm(v):
    """model/training orientation -> DICOM acquisition orientation (inverse of dcm_to_model)."""
    return np.ascontiguousarray(np.swapaxes(v[:, ::-1, :], 1, 2))


def to_zhw_gray(a):
    """Normalize any loaded array to a single-channel (Z, H, W) float32 volume.
    Handles 2D images (H,W)->(1,H,W), RGB(A) (H,W,3/4) or (Z,H,W,3/4)->luminance,
    and other 4D arrays. A genuine 3-slice DBT volume (3,H,W) is left intact."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:                                   # single 2D image
        a = a[None]
    elif a.ndim == 3 and a.shape[-1] in (3, 4) and a.shape[0] not in (3, 4):
        a = a[..., :3].mean(-1)[None]                 # (H,W,3) RGB -> (1,H,W)
    elif a.ndim == 4:
        if a.shape[-1] in (3, 4):                     # (Z,H,W,3) -> (Z,H,W)
            a = a[..., :3].mean(-1)
        else:
            a = a.reshape(a.shape[0], a.shape[1], a.shape[2])
    return np.ascontiguousarray(a)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_nifti(arr, affine, path):
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), affine), path)


def write_dicom_series(vol, outdir, template_file=None, desc="prediction", scale=1):
    """Per-slice Secondary Capture DICOM from a (Z,rows,cols) volume already in
    DICOM acquisition orientation (caller maps with model_to_dcm)."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    os.makedirs(outdir, exist_ok=True)
    tmpl = pydicom.dcmread(template_file, stop_before_pixels=True, force=True) if template_file else None
    series_uid = generate_uid()
    study_uid = str(getattr(tmpl, "StudyInstanceUID", generate_uid())) if tmpl else generate_uid()
    v = (np.clip(vol, 0, None) * scale).astype(np.uint16)
    for i in range(v.shape[0]):
        ds = Dataset()
        for tag in ("PatientID", "PatientName", "StudyDate", "StudyID", "AccessionNumber"):
            if tmpl is not None and tag in tmpl:
                setattr(ds, tag, getattr(tmpl, tag))
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"      # Secondary Capture Image Storage
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.Modality = "MG"
        ds.SeriesDescription = desc
        ds.InstanceNumber = i + 1
        ds.Rows, ds.Columns = int(v.shape[1]), int(v.shape[2])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        if tmpl is not None and "PixelSpacing" in tmpl:
            ds.PixelSpacing = tmpl.PixelSpacing
        ds.PixelData = v[i].tobytes()
        meta = Dataset()
        meta.MediaStorageSOPClassUID = ds.SOPClassUID
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        fds = FileDataset(None, ds, file_meta=meta, preamble=b"\0" * 128)
        fds.is_little_endian = True
        fds.is_implicit_VR = False
        fds.save_as(os.path.join(outdir, f"{i + 1:04d}.dcm"))


def copy_original_dicom(files, outdir):
    os.makedirs(outdir, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(outdir, os.path.basename(f)))
