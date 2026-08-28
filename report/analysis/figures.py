"""Figures that do not exist yet, built from the verified result CSVs only.

Colour policy: damage class is ordinal and already carries tab10 colours in the EDA
figures copied from the notebooks, so nothing here re-colours it. Damage class always
sits on the x axis and colour is spent only on the genuinely categorical dimension
(model / truth-vs-prediction / regression specification), using validated slots.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
# <repo>/report/analysis/<this file>  ->  <repo>
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
HERE = Path(__file__).parent
OUT = ROOT / "NEW_report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SHORT = ["none", "minor", "major", "destroyed"]
SPLITS = ["val", "test_id", "test_ood"]
TITLES = {"val": "val (seen types)", "test_id": "test_id (seen types)",
          "test_ood": "test_ood (wildfire, unseen)"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "grid.color": "#e2e1dd", "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def tidy(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------- 1. per-class F1
pcP = pd.read_csv(R / "pretrained_resnet18/results_per_class.csv")
pcS = pd.read_csv(R / "scratch_cnn/results_per_class.csv")

fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.9), sharey=True)
x, w = np.arange(4), 0.38
for ax, split in zip(axes, SPLITS):
    p = pcP[pcP.split == split].set_index("class").loc[CLASSES, "f1"].to_numpy()
    s = pcS[pcS.split == split].set_index("class").loc[CLASSES, "f1"].to_numpy()
    for off, vals, col, lab in [(-w / 2, p, BLUE, "pretrained ResNet-18"),
                                (w / 2, s, ORANGE, "from scratch")]:
        bars = ax.bar(x + off, vals, w, color=col, label=lab, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.6, color=INK2)
    ax.set(xticks=x, ylim=(0, 1.06), title=TITLES[split])
    ax.set_xticklabels(SHORT, rotation=20, ha="right")
    tidy(ax)
axes[0].set_ylabel("per-class F1")
axes[0].legend(loc="upper center", fontsize=7.5, ncol=1)
fig.tight_layout()
fig.savefig(OUT / "per_class_f1.png", dpi=200)
plt.close(fig)
print("per_class_f1.png")

# ---------------------------------------------------------------- 2. OOD prior shift
p_ood = pcP[pcP.split == "test_ood"].set_index("class").loc[CLASSES]
pred_P = (p_ood.support * p_ood.recall / p_ood.precision).round().to_numpy()
cm = pd.read_csv(R / "scratch_cnn/confusion_matrices.csv")
cm_ood = cm[cm.split == "test_ood"].set_index("true_class").loc[CLASSES, CLASSES].to_numpy()
pred_S = cm_ood.sum(axis=0).astype(float)
true = p_ood.support.to_numpy().astype(float)

fig, ax = plt.subplots(figsize=(5.2, 3.0))
w = 0.27
for off, vals, col, lab in [(-w, true, MUTED, "ground truth"),
                            (0.0, pred_P, BLUE, "predicted, pretrained"),
                            (w, pred_S, ORANGE, "predicted, from scratch")]:
    bars = ax.bar(x + off, vals, w, color=col, label=lab, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.10, f"{int(v):,}",
                ha="center", va="bottom", fontsize=6.2, color=INK2, rotation=90)
ax.set(xticks=x, yscale="log", ylim=(50, 120000), ylabel="patches in test_ood (log)")
ax.set_xticklabels(SHORT)
ax.legend(fontsize=7.2, loc="upper right")
tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "ood_prior_shift.png", dpi=200)
plt.close(fig)
print("ood_prior_shift.png")

# ---------------------------------------------------------------- 3. scratch confusion
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1))
for ax, split in zip(axes, SPLITS):
    m = cm[cm.split == split].set_index("true_class").loc[CLASSES, CLASSES].to_numpy().astype(float)
    norm = m / m.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=x, yticks=x, title=TITLES[split], xlabel="predicted")
    ax.set_xticklabels(SHORT, rotation=20, ha="right")
    ax.set_yticklabels(SHORT)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else INK)
axes[0].set_ylabel("true")
fig.tight_layout()
fig.savefig(OUT / "confusion_scratch.png", dpi=200)
plt.close(fig)
print("confusion_scratch.png")

# ---------------------------------------------------------------- 4. socio gradient
t = pd.read_csv(HERE / "tracts_final.csv")
q = pd.read_csv(HERE / "socio_quintiles.csv")
big = t[t.event.map(t.event.value_counts()) >= 15].copy()
big["q_within"] = big.groupby("event").income_2017.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax = axes[0]
bars = ax.bar(range(5), q.damage, 0.62, color=BLUE, zorder=3)
for bar, v, inc in zip(bars, q.damage, q.med_income):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7, color=INK2)
ax.set(xticks=range(5), ylim=(0, 0.72), ylabel="mean damage score (0 to 3)",
       title="Pooled across all ten events")
ax.set_xticklabels([f"Q{i+1}\n${int(v/1000)}k" for i, v in enumerate(q.med_income)], fontsize=7)
ax.set_xlabel("block-group median household income, quintile")
tidy(ax)

ax = axes[1]
g = big.groupby("q_within", observed=True).damage_score_mean.agg(["mean", "sem", "size"])
bars = ax.bar(range(3), g["mean"], 0.55, color=AQUA, zorder=3)
ax.errorbar(range(3), g["mean"], yerr=1.96 * g["sem"], fmt="none", ecolor=INK2,
            elinewidth=0.9, capsize=3, zorder=4)
for bar, v in zip(bars, g["mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2, v - 0.035, f"{v:.2f}",
            ha="center", va="top", fontsize=7, color="white")
ax.set(xticks=range(3), ylim=(0, 0.72), title="Income tercile computed within each event")
ax.set_xticklabels([f"{k}\nn={int(n)}" for k, n in zip(g.index, g["size"])], fontsize=7)
ax.set_xlabel("relative income inside the same disaster")
tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "socio_income_gradient.png", dpi=200)
plt.close(fig)
print("socio_income_gradient.png")

# ---------------------------------------------------------------- 5. coefficient forest
reg = pd.read_csv(HERE / "socio_regression_final.csv")
specs = ["naive (socioeconomic)", "controlled (+ built environment)", "+ event fixed effects"]
labels = {"income_2017": "median income", "median_home_value_2017": "median home value",
          "renter_pct_2017": "renter share", "building_density": "building density",
          "building_age": "median building age"}
feats = list(labels)
cols = {specs[0]: BLUE, specs[1]: ORANGE, specs[2]: AQUA}
short = {specs[0]: "naive: socioeconomic only", specs[1]: "+ built environment",
         specs[2]: "+ event fixed effects"}

fig, ax = plt.subplots(figsize=(5.6, 3.2))
offs = [0.26, 0.0, -0.26]
for spec, off in zip(specs, offs):
    d = reg[reg.model == spec].set_index("feature")
    ys, cs, es = [], [], []
    for i, f in enumerate(feats):
        if f not in d.index:
            continue
        ys.append(i + off)
        cs.append(d.loc[f, "coef"])
        es.append(1.96 * d.loc[f, "se"])
    ax.errorbar(cs, ys, xerr=es, fmt="o", ms=4.5, color=cols[spec], ecolor=cols[spec],
                elinewidth=1.4, capsize=2.5, label=short[spec], zorder=3)
ax.axvline(0, color=INK2, lw=0.9, zorder=2)
ax.set(yticks=range(len(feats)), xlabel="standardised coefficient on tract damage score",
       ylim=(-0.6, len(feats) - 0.4))
ax.set_yticklabels([labels[f] for f in feats])
ax.invert_yaxis()
ax.legend(fontsize=7.2, loc="lower right")
tidy(ax, grid_axis="x")
fig.tight_layout()
fig.savefig(OUT / "socio_coefficients.png", dpi=200)
plt.close(fig)
print("socio_coefficients.png")
