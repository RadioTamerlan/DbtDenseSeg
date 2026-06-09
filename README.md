# DbtDenseSeg — DBT dense / area / muscle segmentation pipeline

Ensemble inference for Digital Breast Tomosynthesis (DBT): three models
(**breast area**, **pectoral muscle**, **dense tissue**) combined as
`dense ∩ area ∩ ¬muscle`. Accepts **DICOM or NIfTI**, runs on **GPU or CPU**,
and writes the original + predicted masks next to each input.

**Architectures, training, loss functions, and external DBTex performance (with
plots): see [MODEL_CARD.md](MODEL_CARD.md).**

**Model weights (private):** [huggingface.co/RadioTamerlan/DbtDenseSeg-weights](https://huggingface.co/RadioTamerlan/DbtDenseSeg-weights)
— fetched by `get_weights.py` (needs an HF read token).

## Prerequisites (Linux / Windows / macOS)
- **Miniconda or Anaconda** — https://docs.conda.io/en/latest/miniconda.html
- **git**
- Disk: ~3 GB (conda env) + ~1.4 GB (weights)
- GPU optional. With an NVIDIA GPU it uses CUDA automatically; otherwise it runs
  on **CPU** (slower — the 3D model takes minutes/volume and ~8–16 GB RAM).

## Setup

```bash
git clone https://github.com/RadioTamerlan/DbtDenseSeg.git
cd DbtDenseSeg

# 1) create the conda env (named "RadDad") — Linux / Windows / macOS
conda env create -f environment.yml
conda activate RadDad

# 2) set your Hugging Face read token (the weights repo is private), then download
python get_weights.py

# 3) run
python dbtdenseseg/run_pipeline.py --input /path/to/patients --format both
```

### Setting the HF token (step 2) — per shell
The weights repo id is already the default; you just need a **read** token:

| Shell | command |
|---|---|
| Linux / macOS (bash/zsh) | `export HF_TOKEN=hf_xxx` |
| Windows PowerShell | `$env:HF_TOKEN="hf_xxx"` |
| Windows cmd | `set HF_TOKEN=hf_xxx` |

(Same syntax for the optional `DBTDENSESEG_HF_REPO` / `DBTDENSESEG_WEIGHTS` /
`CUDA_VISIBLE_DEVICES` variables.)

## Platform notes
- **Linux / Windows + NVIDIA GPU:** `pip` installs a CUDA PyTorch build; runs on
  GPU with `--device auto`.
- **macOS / no NVIDIA GPU:** runs on **CPU** (`--device cpu` or `auto`). Works,
  but the 3D dense model is slow and memory-heavy.
- **Windows console:** the progress bar auto-switches to ASCII if your terminal
  can't render emoji. For the full ⛏️ bar use Windows Terminal or set
  `PYTHONUTF8=1`.

## Input layout
A root folder of patient subfolders; each series is a NIfTI file **or** a DICOM
series (multi-frame `.dcm` or a folder of per-slice `.dcm`):

```
patients/
  PatientA/
    scan_MLO.nii.gz          # NIfTI series
    seriesX/ *.dcm           # OR a DICOM series
```

View (CC vs MLO/ML) comes from DICOM `ViewPosition` or a `_CC`/`_MLO` token in
the name; it only gates the muscle model (run on MLO/ML, skipped on CC).

## Preprocessing DICOM (optional — recommended for messy data)
If your DICOMs are **split into one file per slice**, or the **view is missing
from `ViewPosition`**, run this first. It assembles each series into a single
NIfTI and **bakes the detected view into the file name**:

```bash
python dbtdenseseg/preprocess_dicom.py --input <dicom root> --out <nifti root>
python dbtdenseseg/run_pipeline.py     --input <nifti root> --format both
```
View + laterality are detected from **many** headers (`ViewPosition`,
`ViewCodeSequence`, `SeriesDescription`/`ProtocolName`/…), then the folder/file
name, then image content — and written into the filename (e.g. `…_L_MLO.nii.gz`),
so the pipeline reads the view straight from the name. `--default-view CC` sets a
fallback when nothing is found.

## Output
For each series → `<series folder>/model prediction/`:
```
<name>_original.(nii.gz | _dicom/)
<name>_ensemble_mask.(nii.gz | _dicom/)     # main result: dense ∩ area ∩ ¬muscle
<name>_dense_mask.*  <name>_area_mask.*  <name>_muscle_mask.*
```
`--format` selects `nii` / `dcm` / `both`.

## Percent density (PD)
After running the pipeline (with `--format nii` or `both`), compute breast percent
density — analogous to TomoLIBRA's VBD:

```bash
python dbtdenseseg/calculate_pd.py --input /path/to/patients --out density.csv
```
`PD% = 100 · dense / breast`, where **dense** = ensemble mask and
**breast** = area ∩ ¬muscle (pectoral muscle excluded). Spacing cancels, so PD
needs no calibration. Absolute dense volume (**ADV**, cm³) is also written when the
NIfTI carries real voxel spacing (default volumes use 1×1×1 placeholder spacing, so
ADV is in placeholder units there). Output columns: `patient, series, dense_voxels,
breast_voxels, PD_percent, voxel_volume_mm3, ADV_cm3`.

## Options
| flag | default | meaning |
|---|---|---|
| `--input` | (required) | root folder of patient subfolders |
| `--format` | `nii` | `nii` / `dcm` / `both` |
| `--device` | `auto` | `auto` (GPU else CPU) / `cuda` / `cpu` |
| `--threshold` | `0.5` | probability threshold for all masks |
| `--harmonize` | off | intensity histogram-match to a Hologic reference (GE/Siemens) |
| `--ref` | — | `hologic_reference.npz` (required with `--harmonize`) |

Env vars: `DBTDENSESEG_WEIGHTS` (weights folder), `CUDA_VISIBLE_DEVICES` (pick GPU).

## Notes
- Models were trained on **Hologic** DBT. DICOM is auto-reoriented to the training
  plane; for **GE/Siemens** use `--harmonize` (see `dbtdenseseg/harmonize_dbt.py`).
- DICOM masks are written as **Secondary Capture** (overlay-able), not DICOM-SEG.
- Requires internet on first run to fetch the SegFormer-B2 base from Hugging Face.

## Layout
```
dbtdenseseg/   run_pipeline.py, preprocess_dicom.py, calculate_pd.py, inference.py, io_volume.py, minerbar.py,
               harmonize_dbt.py, models2d.py, models3d.py, dbt_seg_lib.py, area2d_lib.py
weights/       area.pt, muscle.pt, dense.pt  (downloaded, not in git)
environment.yml / requirements.txt / get_weights.py
```
