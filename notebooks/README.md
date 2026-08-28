# EDA & data-split notebooks (local run)

Sections 1–5 of `artur_xbd_NESH_CAN_RUN.ipynb`, split into two standalone notebooks that run
**locally, without Kaggle**. Neither needs torch or a GPU.

| notebook | source sections | writes |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | 1–4 — setup, dataset inspection, EDA, cleaning & quality diagnostics | `results/01_eda/` |
| [`02_data_split.ipynb`](02_data_split.ipynb) | 5 — split audit, leakage checks, event-stratified scene sampling | `results/02_data_split/` |

They write to their own output directories so they never collide with
[`data_cleaning_eda.ipynb`](data_cleaning_eda.ipynb), which owns `results/eda/`.

Run `01_eda.ipynb` first: it caches the label scan that `02` reads. `02` rebuilds the cache itself
if it is absent, so either order works — the first one is just faster.

```powershell
pip install numpy pandas matplotlib pillow pyarrow notebook
jupyter lab            # started from the project root
```

If you start Jupyter somewhere else, set `PROJECT_ROOT_OVERRIDE` in the PATHS cell. Nothing else
needs editing.

## Data layout

Both notebooks read `data/splits.csv` (the team split index from
[`../scripts/prepare_splits.py`](../scripts/prepare_splits.py)) and rebuild every image/label path
from `(origin, subdir, name)` — never from the `path` column and never from a directory scan,
because the Kaggle mirror ships a partial `xbd/train/` that duplicates `tier1` and is absent from
`splits.csv`; globbing would place the same scene in two splits.

The layout in use here is the Kaggle `xbd-dataset` archive extracted under `data_same/`:

```
<project root>/
    data/
        splits.csv                 <- split index only; no imagery lives here
    data_same/
        xbd/{hold,test,tier1,tier3}/{images,labels,masks}
```

Two other layouts work with no edits, because the resolver searches `data_same/`, then `data/`,
then the project root: `xbd/` directly in the project root (see
[`../docs/local_setup.md`](../docs/local_setup.md)), and the per-archive form
`data/{train,test,tier3,hold}/` where the xView2 `train` tar IS xBD `tier1`.

The `masks/` directory (called `targets/` in the xView2 tars) holds **segmentation** labels. This
per-building **classification** task never reads it.

## Verification — reproduces the Kaggle reference run exactly

Executed end to end on 2026-08-28 against the complete 11,034-scene dataset. Every published
reference figure came back identical:

| quantity | reference | this run |
|---|---|---|
| scenes / building polygons | 11,034 | 11,034 · 425,368 |
| train class mix (%) | 72.0 / 9.7 / 8.4 / 6.2 | 72.04 / 9.65 / 8.41 / 6.22 |
| test_ood class mix (%) | 80.2 / 0.9 / 0.9 / 14.7 | 80.18 / 0.91 / 0.94 / 14.67 |
| non-fire vs wildfire mix | 70.2/9.9/7.2/6.5 vs 78.8/1.2/1.1/14.8 | identical |
| empty scenes, test_ood | 54.1%, median 0 | 54.1%, median 0 |
| footprint median / p90 / p99 px | 35 / 64 / 349 | 35.0 / 63.9 / 349.1 |
| sub-8px footprints | 3.51% | 3.51% |
| cleaning cost | 6.7% of polygons | 6.7% (396,899 kept) |
| occlusion flags | 11 train, 3 val, 0 test | 11 train, 3 val, 0 test |
| occlusion ρ (edges) | −0.366 | −0.366 |
| near-duplicate pairs | 28, 19 cross-split | 28, 19 cross-split |
| eligible scenes | 2,393 / 293 / 910 / 1,990 | identical |
| patches from scenes | 135,262 from 1,698 | 135,262 from 1,698 |
| patches per split | 72,916 / 20,446 / 27,273 / 14,627 | identical |
| train class weights | 0.33 / 2.38 / 2.91 / 4.92 | 0.330 / 2.379 / 2.910 / 4.919 |

## Running on a partial dataset

If only some archives are extracted, the coverage cell (§2.2) drops the scenes whose files are
absent, prints exactly what is missing, and continues. Every cache file is keyed on a coverage tag
(`work/cache/buildings__hold-test-tier1-tier3_11034.parquet`), so a partial scan can never be
silently reused as if it were the full one once the rest of the data arrives.

A partial run still reproduces the qualitative findings, but not the numbers above. Prose in the
notebooks that quotes exact figures is marked *"reference run"*.

## Outputs

`work/` holds derived data and is gitignored: `work/cache/` (parquet scans, so a re-run skips the
slow steps) and `work/split/patch_manifest.csv` (~66 MB, one row per building, regenerable in
seconds from `results/02_data_split/selected_scenes.csv`).
