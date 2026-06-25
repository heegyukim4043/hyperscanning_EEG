# Triadic Hyperscanning EEG Dataset and MTAD-GAT Processing Pipeline

This folder contains raw triadic hyperscanning EEG session files and Python scripts for generating model-ready PKL files for cooperation detection.

## Dataset Structure

```text
data_dl/
|-- G01/
|   |-- G01_session1_raw.mat
|   |-- G01_session2_raw.mat
|   |-- G01_session3_raw.mat
|   `-- G01_session4_raw.mat
|-- ...
|-- G11/
|   |-- G11_session1_raw.mat
|   |-- G11_session2_raw.mat
|   |-- G11_session3_raw.mat
|   `-- G11_session4_raw.mat
|-- model_scripts/
|-- inspect_data_dl_raw.py
|-- build_processed_from_raw.py
|-- smoke_test_processed_inputs.py
|-- requirements.txt
`-- requirements_data_dl_pipeline.txt
```

Expected raw data:

- `11` groups: `G01` to `G11`
- `4` sessions per group
- `44` raw MATLAB v7.3/HDF5 `.mat` files
- each session contains continuous EEG and session-level cooperation labels

Expected core variables in each `.mat` file:

| Variable | Expected shape | Meaning |
|---|---:|---|
| `data` | `time x 57` | 19 EEG channels x 3 participants |
| `srate` | scalar | sampling rate, expected `300 Hz` |
| `decision_results_session` | `3 x 10` | pair-wise cooperation labels for 10 trials |
| `event_marker` | `1 x N` | event marker codes |
| `event_latency_samples` | `1 x N` | sample index for each event marker |
| `event_timestamp` | `1 x N` | timestamp in seconds for each event marker |
| `chanlocs` | `19 x 1` | EEG channel locations/names |

## Label Definition

The generated label is a binary matrix with shape:

```text
time x 3
```

Columns:

| Column | Pair | Meaning |
|---:|---|---|
| 1 | pair12 | participant 1 and participant 2 cooperation |
| 2 | pair13 | participant 1 and participant 3 cooperation |
| 3 | pair23 | participant 2 and participant 3 cooperation |

The default positive event window is `0-6 s` from the trial decision marker. Non-event periods are labeled `0`.

Event timing is defined by matched event arrays. For each event index `i`:

```text
event_marker(i) occurs at event_latency_samples(i) samples
event_marker(i) occurs at event_timestamp(i) seconds
```


## Preprocessing

`build_processed_from_raw.py` applies:

1. Bandpass filtering: `1-55 Hz`
2. Notch filtering: `60 Hz`
3. Optional Infomax ICA + ICLabel rejection
   - components labeled `eye blink` or `muscle artifact`
   - probability threshold default: `0.9`
   - enabled only with `--use_ica`

ICA/ICLabel is optional because it requires extra dependencies and is computationally heavier.

## Install Dependencies

```powershell
pip install -r data_dl\requirements.txt
```

If you need a CUDA-specific PyTorch/DGL build, install the matching `torch` and `dgl` wheels for your CUDA version first, then install the remaining packages from `requirements.txt`.

## Step 1: Inspect Raw Files

```powershell
python data_dl\inspect_data_dl_raw.py --raw_root data_dl --max_files 3
```

This checks:

- whether all `44` raw files exist
- header format
- HDF5 variable names
- dataset shapes and dtypes

## Step 2: Dry-Run Preprocessing

```powershell
python data_dl\build_processed_from_raw.py `
  --raw_root data_dl `
  --out_dir data_dl\processed_pkl `
  --dry_run
```

This loads raw sessions, filters EEG, builds labels, and reports shapes/positive rates without writing output files.

## Step 3: Build Model-Ready PKL Files

```powershell
python data_dl\build_processed_from_raw.py `
  --raw_root data_dl `
  --out_dir data_dl\processed_pkl `
  --overwrite
```

Optional ICA/ICLabel version:

```powershell
python data_dl\build_processed_from_raw.py `
  --raw_root data_dl `
  --out_dir data_dl\processed_pkl_ica `
  --use_ica `
  --overwrite
```

Generated output follows the existing MTAD-GAT input contract:

```text
machine-G-S_train.pkl
machine-G-S_test.pkl
machine-G-S_train_label_vec.pkl
machine-G-S_test_label_vec.pkl
```

For each group/session fold:

```text
test = held-out session S
train = remaining 3 sessions concatenated
```

## Step 4: Smoke-Test Training Inputs

```powershell
python data_dl\smoke_test_processed_inputs.py `
  --processed_dir data_dl\processed_pkl
```

This verifies:

- `44 / 44` folds exist
- data shape is `[T, 57]`
- label shape is `[T, 3]`
- data/label time lengths match
- labels are binary `0/1`

## Step 5: Train Existing Models

The main bundled model entrypoint is:

```text
data_dl/model_scripts/train_PD3_interonly_coop3_dgcn_fb.py
```

Example focal-loss run:

```powershell
python data_dl\model_scripts\train_PD3_interonly_coop3_dgcn_fb.py `
  --processed_dir data_dl\processed_pkl `
  --subject_range 1,11 --sub_range 1,4 `
  --label_suffix label_vec `
  --use_cuda `
  --downsample 2 `
  --lookback 300 `
  --scaling robust_mad `
  --use_gatv2 `
  --decoder transformer `
  --lambda_coop 1.0 `
  --lambda_align 0.0 `
  --lambda_delta 0.0 `
  --lambda_coop_warmup_boundaries 30,60 `
  --lambda_coop_warmup_values 0.0,0.3,1.0 `
  --epochs 200 `
  --bs 64 `
  --patience 30 `
  --min_delta 1e-3 `
  --focal_gamma 2.0 `
  --pos_weight_mode dynamic `
  --run_root runs_data_dl_interonly_coop3_clean_focal2 `
  --seed 2026
```

## Bundled Model/Analysis Scripts

`model_scripts/` includes the main model and analysis files needed to restart the current inter-brain MTAD-GAT workflow:

- `train_PD3_interonly_coop3_dgcn_fb.py`
- `mtad_gat_interonly_coop3_dgcn_fb.py`
- `training_coop.py`
- `dgl_layers_inter3.py`
- `modules.py`
- `module_transformer.py`
- `module_snn_decoder.py`
- `eval_coop_head.py`
- `prediction.py`
- `eval_methods.py`
- `spot.py`
- `utils.py`
- `utils_PD.py`
- `analyze_plv_bands.py`
- `compare_plv_attn.py`

## Notes for GitHub Upload

- Total folder size is approximately `560 MB`.
- Individual raw files are below GitHub's `100 MB` per-file limit.
- Git LFS is recommended for raw `.mat` files to avoid a large Git history.
- Generated files such as `processed_pkl/`, model runs, and `__pycache__/` are ignored by `.gitignore`.

## Citation and Use

Add citation, data-use, and license terms here before public release.

