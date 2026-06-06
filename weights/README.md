# Weights

The pipeline needs three checkpoints here:

```
weights/area.pt      # SegFormer-B2 breast-area  (~314 MB)
weights/muscle.pt    # SegFormer-B2 pectoral muscle (~314 MB)
weights/dense.pt     # SwinUNETR 3D dense tissue  (~720 MB)
```

They are **not** stored in git (too large). Get them one of two ways:

**A. Hugging Face (recommended)**
```bash
export DBTDENSESEG_HF_REPO="RadioTamerlan/DbtDenseSeg-weights"
export HF_TOKEN="hf_xxx"      # only if the HF repo is private
python ../get_weights.py
```

**B. Manual** — copy `area.pt`, `muscle.pt`, `dense.pt` into this folder.

To point at a different folder instead, set `DBTDENSESEG_WEIGHTS=/path/to/folder`.
