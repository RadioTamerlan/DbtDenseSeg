"""Percent density (PD) from DbtDenseSeg outputs — analogous to TomoLIBRA's VBD.

    PD% = 100 · (dense voxels) / (breast voxels)

      dense  = ensemble mask  (dense ∩ area ∩ ¬muscle)
      breast = area ∩ ¬muscle (breast tissue with pectoral muscle excluded)

Spacing cancels in the ratio, so PD needs no calibration (works even with the
1×1×1 placeholder spacing). If the NIfTI carries *real* voxel spacing, the
absolute dense volume (ADV, cm³ = dense_voxels · voxel_volume) is also reported —
otherwise ADV is in placeholder units (flagged).

Run it on a folder that the pipeline has already populated with masks:

    python calculate_pd.py --input <patients root> --out density.csv

It reads  <...>/model prediction/<stem>_{ensemble,area,muscle}_mask.nii.gz .
"""
import argparse
import csv
import glob
import os
import numpy as np
import nibabel as nib

PRED_DIR = "model prediction"
SUFFIX = "_ensemble_mask.nii.gz"


def load(p):
    return np.asarray(nib.load(p).dataobj)


def main():
    ap = argparse.ArgumentParser(description="Percent density (PD) from DbtDenseSeg masks")
    ap.add_argument("--input", required=True, help="root folder scanned for 'model prediction' outputs")
    ap.add_argument("--out", default="density.csv")
    args = ap.parse_args()

    ens_files = sorted(glob.glob(os.path.join(args.input, "**", PRED_DIR, "*" + SUFFIX),
                                 recursive=True))
    if not ens_files:
        raise SystemExit(f"No '*{SUFFIX}' under {args.input} (run the pipeline first, "
                         f"--format nii or both).")

    rows = []
    placeholder = False
    for ens_p in ens_files:
        d = os.path.dirname(ens_p)
        stem = os.path.basename(ens_p)[:-len(SUFFIX)]
        area_p = os.path.join(d, stem + "_area_mask.nii.gz")
        musc_p = os.path.join(d, stem + "_muscle_mask.nii.gz")
        if not (os.path.isfile(area_p) and os.path.isfile(musc_p)):
            print(f"  skip (missing area/muscle mask): {stem}")
            continue
        dense = load(ens_p) > 0
        breast = (load(area_p) > 0) & ~(load(musc_p) > 0)
        dv, bv = int(dense.sum()), int(breast.sum())
        pd = 100.0 * dv / bv if bv else float("nan")
        zooms = nib.load(ens_p).header.get_zooms()[:3]
        vox_mm3 = float(np.prod(zooms))
        if abs(vox_mm3 - 1.0) < 1e-6:
            placeholder = True
        adv_cm3 = dv * vox_mm3 / 1000.0
        patient = os.path.relpath(d, args.input).split(os.sep)[0]
        rows.append(dict(patient=patient, series=stem, dense_voxels=dv, breast_voxels=bv,
                         PD_percent=round(pd, 2), voxel_volume_mm3=round(vox_mm3, 5),
                         ADV_cm3=round(adv_cm3, 3)))
        print(f"  {patient}/{stem}: PD={pd:5.1f}%   dense={dv}  breast={bv}")

    cols = ["patient", "series", "dense_voxels", "breast_voxels", "PD_percent",
            "voxel_volume_mm3", "ADV_cm3"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} series -> {args.out}")
    if rows:
        pds = [r["PD_percent"] for r in rows if r["PD_percent"] == r["PD_percent"]]
        if pds:
            print(f"mean PD = {np.mean(pds):.1f}%  (n={len(pds)})")
    if placeholder:
        print("NOTE: voxel spacing is the 1×1×1 placeholder for some series -> ADV_cm3 is "
              "NOT real cm³ (PD% is unaffected). For real ADV, supply NIfTIs with true spacing.")


if __name__ == "__main__":
    main()
