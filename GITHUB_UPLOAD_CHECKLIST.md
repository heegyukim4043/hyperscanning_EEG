# GitHub Upload Checklist

## Before Upload

- Confirm all `44` raw files are present:

```powershell
python data_dl\inspect_data_dl_raw.py --raw_root data_dl --max_files 1
```

- Confirm raw `.mat` files open as MATLAB v7.3/HDF5.
- Confirm `README.md` has the intended citation/license text.
- Decide whether raw `.mat` files should use Git LFS.

## Recommended Git LFS

The raw files are below `100 MB` each, but the folder is about `560 MB`.

Recommended:

```powershell
git lfs install
git lfs track "data_dl/**/*.mat"
git add .gitattributes
```

Then add:

```powershell
git add data_dl
git commit -m "Add triadic hyperscanning EEG data pipeline"
git push
```

## Files to Include

- `G01/` to `G11/`
- `README.md`
- `requirements_data_dl_pipeline.txt`
- `inspect_data_dl_raw.py`
- `build_processed_from_raw.py`
- `smoke_test_processed_inputs.py`
- `model_scripts/`

## Files Not to Include

The local `.gitignore` excludes:

- generated `processed_pkl/`
- generated run folders
- Python cache files
- model checkpoints

## Post-Clone Smoke Test

After cloning on another machine:

```powershell
pip install -r data_dl\requirements_data_dl_pipeline.txt
python data_dl\inspect_data_dl_raw.py --raw_root data_dl --max_files 3
python data_dl\build_processed_from_raw.py --raw_root data_dl --out_dir data_dl\processed_pkl --dry_run
```

If the dry-run passes, build the processed PKLs:

```powershell
python data_dl\build_processed_from_raw.py --raw_root data_dl --out_dir data_dl\processed_pkl --overwrite
python data_dl\smoke_test_processed_inputs.py --processed_dir data_dl\processed_pkl
```
