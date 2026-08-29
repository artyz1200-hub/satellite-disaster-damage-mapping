# Mapping Natural-Disaster Damage from Satellite Imagery

Per-building damage classification from paired pre/post satellite imagery on
xBD. The central question is cross-disaster transfer: both models train without
wildfire and are evaluated on wildfire as an unseen test-OOD domain.

**Team:** Bhuvanesh Dinesh Wadhwani · Artur Zavistovskyi · Anastasiia Khitrova · Yitian Zhou

**Final report:** [`report/main.tex`](report/main.tex) — compile on Overleaf, see
[`report/README.md`](report/README.md).

## Experiment

Both systems use the same 135,262 paired 64x64 patches and the same scene-level
train/val/test-ID/test-OOD split.

1. A compact shared-weight Siamese CNN trained from random initialization.
2. A shared-weight Siamese ResNet-18 initialized from ImageNet, then fine-tuned.

Each branch encodes one of the pre/post patches. The classifier receives
`[f_pre, f_post, abs(f_pre - f_post)]` and predicts no damage, minor damage,
major damage, or destroyed.

## Results

Fifty epochs was the maximum; both runs used early stopping on validation
macro-F1 with patience 8.

| Model | Epochs run | Best epoch | Val macro-F1 | Test-ID macro-F1 | Test-OOD macro-F1 |
|---|---:|---:|---:|---:|---:|
| ImageNet-pretrained Siamese ResNet-18 | 43 | 35 | 0.6656 | 0.6498 | 0.4292 |
| From-scratch Siamese CNN | 16 | 8 | 0.5967 | 0.6148 | 0.3768 |

The headline macro-F1 drop is misleading on its own. Out of domain the
pretrained model scores **higher** on `destroyed` than it does in domain (F1
0.758 against 0.687), `no-damage` precision reaches 0.966, and threshold-free
ROC-AUC holds at 0.814. The collapse is confined to `minor` and `major`, which
are 19.1% of training patches but only 1.7% of wildfire patches. The features
transfer; the decision calibration does not. Section 7 of the report develops
this with four independent measurements.

See [`docs/experiment_results.md`](docs/experiment_results.md) and
[`results/model_comparison.csv`](results/model_comparison.csv) for the full
comparison and reproducibility notes.

## Trained model downloads

The two best checkpoints are too large for normal Git history and are published
as a GitHub Release instead:

- [`pretrained_siamese_resnet18.pt`](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/download/v1.0-trained-models/pretrained_siamese_resnet18.pt)
  (44.23 MB): ImageNet-pretrained Siamese ResNet-18, best epoch 35.
- [`from_scratch_siamese_cnn.pt`](https://github.com/artyz1200-hub/satellite-disaster-damage-mapping/releases/download/v1.0-trained-models/from_scratch_siamese_cnn.pt)
  (5.25 MB): compact Siamese CNN trained from scratch, best epoch 8.

The assets are published in this repository's `v1.0-trained-models` Release.
They are versioned model artifacts, not Git history, and both links are used
only for checkpoint download.

## Repository layout

```text
.
├── report/                    final report: LaTeX source, figures, number-checking scripts
│   ├── main.tex               the report (compile this)
│   ├── figures/               19 figures; main.tex references them as figures/<name>.png
│   └── analysis/              scripts that produced every number quoted in the report
├── notebooks/                 EDA, split, and the two training notebooks
├── scripts/                   dataset download and split-index generation
├── data/                      split definitions and census covariates (not xBD imagery)
├── docs/                      assignment brief, specification, setup and results guides
├── results/                   metrics, tables and plots written by the notebooks
├── requirements.txt           Python dependencies
├── README.md                  project entry point
├── .gitignore                 excludes xBD imagery, work dirs, checkpoints, LaTeX build files
└── LICENSE                    MIT license
```

### Report

| Path | Purpose |
|---|---|
| `report/main.tex` | The final report. Two-column, pdfLaTeX, no `.bib` and no non-default packages. |
| `report/README.md` | How to compile on Overleaf, the estimated page count, and what to cut first if it overflows the 10-page limit. |
| `report/figures/` | All 19 figures. Eleven are referenced by `main.tex`; the rest are supplementary EDA plots kept for reference. |
| `report/analysis/verify.py` | Recomputes every number quoted in the report from the CSVs in `results/`. |
| `report/analysis/collect.py`, `figures.py`, `socio.py` | Assemble the result tables, render the report figures, and run the Part 3 regressions. |
| `report/analysis/verified_numbers.json` | The verified values, so a claim in the text can be traced to its source. |

### Notebooks

| Path | Purpose |
|---|---|
| `notebooks/01_eda.ipynb` | Full-dataset exploratory analysis and the cleaning ledger, run locally without a GPU. Writes `results/01_eda/`. |
| `notebooks/02_data_split.ipynb` | Split audit, leakage checks, and the event-stratified scene sampling that defines the patch set. Writes `results/02_data_split/`. |
| `notebooks/README.md` | Data layout, partial-dataset behaviour, and the verification table showing these two notebooks reproduce the reference run exactly. |
| `notebooks/data_cleaning_eda.ipynb` | Cleans the indexed buildings, performs the modelling EDA, and creates the shared patch arrays under `work-pretrained/patches/`. Writes `results/eda/`. |
| `notebooks/pretrained_resnet18_train.ipynb` | Trains the ImageNet-pretrained Siamese ResNet-18 or loads the published checkpoint for evaluation. |
| `notebooks/train_scratch_cnn.ipynb` | Trains, validates, checkpoints, and evaluates the compact Siamese CNN on the same shared patch arrays. |
| `notebooks/socio_analysis.ipynb` | Part 3: joins damage labels to census block groups and runs the impact regressions. |

### Scripts

| Path | Purpose |
|---|---|
| `scripts/download_xbd.py` | Resumable range downloader for the Kaggle xBD archive. |
| `scripts/prepare_splits.py` | Scans xBD, creates the scene-level train/val/test-ID/test-OOD assignment, and checks leakage and file integrity. |
| `scripts/download_acs.py` | Fetches the American Community Survey covariates used in Part 3. |
| `scripts/download_tiger.py` | Fetches the TIGER/Line census block-group boundaries. |
| `scripts/setup_windows.ps1` | Waits for a download, validates and extracts the archive, and checks the split on Windows. |

### Data definitions

| Path | Purpose |
|---|---|
| `data/splits.csv` | File-level xBD index: 77,238 rows mapping each image, label, and mask file to a scene, disaster, origin, and split. |
| `data/splits_summary.csv` | Scene-level summary: 11,034 rows used to inspect event composition and split counts. |
| `data/acs.csv` | American Community Survey 5-year estimates (2017) per census block group. |
| `data/hazard_exposure.csv` | Hazard-exposure proxy. Retained for provenance; the report explains why it is unusable (identically zero — no hazard layer was ever joined). |

These CSVs define the shared experiment split and the Part 3 covariates. They
are not the xBD image dataset and are safe to keep in Git.

### Documentation

| Path | Purpose |
|---|---|
| `docs/assignment_brief.md` | The original project description, required steps, and grading criteria. |
| `docs/project_specification.md` | Expanded project specification. |
| `docs/experiment_results.md` | Final two-model comparison, interpretation, limitations, and reproduction commands. |
| `docs/local_setup.md` | Windows, Kaggle, CUDA, directory-layout, and local-run instructions. |
| `docs/socio_analysis_README.md` | Part 3 data sources and method. |

### Results

Everything under `results/` is written by a notebook and is the evidence behind
the report's tables and figures.

| Path | Purpose |
|---|---|
| `results/model_comparison.csv` | Single summary table comparing both models across validation, test-ID, and wildfire test-OOD. |
| `results/01_eda/` | Full-dataset EDA and cleaning artefacts: class balance per split and per event, footprint geometry, capture-geometry domain shift, occlusion diagnostic and candidate list, near-duplicate check, cleaning ledger. |
| `results/02_data_split/` | Split composition after cleaning, eligible-versus-selected event mix within each split, patch class balance, and the train-only class weights. |
| `results/eda/` | Artefacts from the earlier combined cleaning + EDA notebook. |
| `results/pretrained_resnet18/` | Overall and per-class metrics, epoch-by-epoch history, and training curves for the pretrained model. |
| `results/scratch_cnn/` | The same for the from-scratch CNN, plus confusion matrices and `run_config.json` (GPU, seed, batch size, stopping rule, parameter count, best epoch). |
| `results/socio_analysis/` | Part 3: per-building and per-tract joins, regression results, and the coefficient and income-gradient plots. |

## Data

xBD is **not committed** — it is roughly 35 GB. Obtain it from
<https://xview2.org> or the Kaggle mirror `qianlanzz/xbd-dataset`, and extract so
that the four origin directories sit together:

```text
<repo>/
  data/splits.csv
  xbd/{hold,test,tier1,tier3}/{images,labels,masks}/
```

`notebooks/01_eda.ipynb` and `02_data_split.ipynb` also accept
`data_same/xbd/...` or the per-archive form `data/{train,test,tier3,hold}/`,
where the xView2 `train` tar is xBD `tier1`. They rebuild every path from
`(origin, subdir, name)` rather than globbing, because the Kaggle mirror ships a
partial `xbd/train/` that duplicates `tier1` and is absent from `splits.csv`.

If only part of the dataset is present the notebooks report exactly what is
missing and continue on what is there; see `notebooks/README.md`.

## Local run

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\jupyter-lab.exe
```

Start Jupyter from the repository root so relative paths resolve. Run
`notebooks/01_eda.ipynb` and `02_data_split.ipynb` for the report EDA and split
audit, then run `data_cleaning_eda.ipynb` to create the shared patch arrays.
After that the two model notebooks can be run independently.

Non-interactively:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\pretrained_resnet18_train.ipynb `
  --output pretrained_resnet18_train.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.timeout=-1
```

The pretrained notebook defaults to checkpoint evaluation; set
`RUN_TRAINING = True` for the full 5+45 epoch schedule. The scratch notebook
supports `RESUME = True` after an interrupted training run.

Seed 42 throughout. The reported runs used an NVIDIA RTX 4070 Laptop GPU.

## Contributions

- **Artur Zavistovskyi** — exploratory analysis, data cleaning and quality
  diagnostics, patch pipeline, pretrained model, evaluation, final report.
- **Bhuvanesh Dinesh Wadhwani** — dataset indexing and the scene-level
  train/val/test-ID/test-OOD split design used by every model in the project.
- **Yitian Zhou** — pretrained model (joint).
- **Anastasiia Khitrova** — from-scratch CNN baseline and the impact and
  recovery analysis (Part 3).

## License

MIT — see [`LICENSE`](LICENSE).
