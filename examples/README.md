# Example

Arrange your data as patient subfolders under one root, then point `--input` at it:

```
examples/data/
  Patient001/
    left_MLO.nii.gz
    right_CC/  *.dcm
```

```bash
conda activate RadDad
python ../dbtdenseseg/run_pipeline.py --input data --format both
```

Results appear in each series' `model prediction/` subfolder. `examples/data/` is
git-ignored, so put test cases there freely.
