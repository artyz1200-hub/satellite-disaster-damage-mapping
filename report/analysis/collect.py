"""Regenerate the patch-grid figure from the current manifest, then collect every
figure the report cites into NEW_report/figures/ under stable names."""
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

mpl.use("Agg")
# <repo>/report/analysis/<this file>  ->  <repo>
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "NEW_report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PATCH_SIZE, BBOX_PAD, SEED = 64, 10, 42
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def load_patch(box, full, pad=BBOX_PAD, size=PATCH_SIZE):
    minx, miny, maxx, maxy = box
    W, H = full.size
    l, t = max(0, int(minx) - pad), max(0, int(miny) - pad)
    r, b = min(W, int(maxx) + pad), min(H, int(maxy) + pad)
    if r - l < 2 or b - t < 2:
        return None
    return np.asarray(full.crop((l, t, r, b)).resize((size, size), Image.BILINEAR), np.uint8)


man = pd.read_csv(ROOT / "work/split/patch_manifest.csv")
rng = np.random.RandomState(SEED)
fig, axes = plt.subplots(4, 8, figsize=(13, 7.0))
cache = {}
for c, cls in enumerate(CLASSES):
    sub = man[(man.split == "train") & (man.subtype == cls)]
    picks = sub.sample(8, random_state=SEED)
    for j, row in enumerate(picks.itertuples(index=False)):
        ax = axes[c, j]
        ax.axis("off")
        for key in (row.img_pre, row.img_post):
            if key not in cache:
                cache[key] = Image.open(key).convert("RGB")
        box = (row.minx, row.miny, row.maxx, row.maxy)
        a = load_patch(box, cache[row.img_pre])
        b = load_patch(box, cache[row.img_post])
        gap = np.full((PATCH_SIZE, 3, 3), 255, np.uint8)
        ax.imshow(np.concatenate([a, gap, b], axis=1))
        if j == 0:
            ax.set_ylabel(cls, fontsize=9)
            ax.axis("on")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
fig.suptitle("Extracted training patches: pre-disaster | post-disaster, one row per damage class",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "patches_grid.png", dpi=170)
plt.close(fig)
print(f"patches_grid.png  (decoded {len(cache)} tiles)")

COPY = {
    # Part 1: exploratory analysis, from the local full-dataset run (results/01_eda)
    "results/01_eda/damage_class_per_split.png": "class_distribution_by_split.png",
    "results/01_eda/damage_composition_per_event_heatmap.png": "class_composition_by_event.png",
    "results/01_eda/empty_scenes_per_event.png": "buildings_per_scene.png",
    "results/01_eda/training_footprint_geometry.png": "footprint_geometry.png",
    "results/01_eda/capture_geometry_domain_shift.png": "capture_geometry.png",
    "results/01_eda/training_examples_by_class.png": "examples_train.png",
    "results/01_eda/ood_examples_by_class.png": "examples_ood.png",
    "results/01_eda/unclassified_rate_by_event.png": "unclassified_rate_by_event.png",
    # Part 1: cleaning diagnostics
    "results/01_eda/occlusion_proxy_correlation.png": "occlusion_correlation.png",
    "results/01_eda/occlusion_gallery.png": "occlusion_gallery.png",
    "results/02_data_split/composition_per_split_after_cleaning.png": "composition_after_cleaning.png",
    # Part 2: training
    "results/pretrained_resnet18/training_curves.png": "training_curves_pretrained.png",
    "results/scratch_cnn/training_curves.png": "training_curves_scratch.png",
}
for src, dst in COPY.items():
    shutil.copy2(ROOT / src, OUT / dst)
    print(f"{dst:38s} <- {src}")

print("\nfigures now in NEW_report/figures:")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name:38s} {p.stat().st_size/1024:8.0f} KB")
