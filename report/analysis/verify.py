"""Verify every number that will go into the final report, straight from results/*.csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# <repo>/report/analysis/<this file>  ->  <repo>
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = {}


def sec(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)


# ------------------------------------------------------------------ Part 1
sec("1. DATASET / SPLITS")
scene = pd.read_csv(R / "01_eda/scene_summary.csv")
print(scene.to_string(index=False))
print("total scenes", scene.scenes.sum(), "total polygons", scene.total_buildings.sum())
OUT["scenes_total"] = int(scene.scenes.sum())
OUT["polygons_total"] = int(scene.total_buildings.sum())

counts = pd.read_csv(R / "01_eda/damage_class_counts_all_splits.csv")
print("\n", counts.to_string(index=False))
comp = pd.read_csv(R / "01_eda/damage_class_composition_per_split.csv")
print("\n", comp.to_string(index=False))

# imbalance ratio in train (excluding un-classified)
tr = counts.set_index("split").loc["train"]
ratio = tr["no-damage"] / tr["destroyed"]
print(f"\ntrain no-damage / destroyed ratio = {ratio:.2f}x")
OUT["train_imbalance_ratio"] = round(float(ratio), 1)

fire = pd.read_csv(R / "01_eda/damage_composition_fire_vs_nonfire.csv")
print("\n", fire.to_string(index=False))

# ------------------------------------------------------------------ cleaning
sec("2. CLEANING LEDGER")
led = pd.read_csv(R / "01_eda/cleaning_ledger.csv")
print(led[["rule", "scope", "removed", "kept", "threshold"]].to_string(index=False))
unc = 14011
tiny = 14458
print(f"\nun-classified   {unc} / 425368 = {100*unc/425368:.2f}%")
print(f"tiny footprints {tiny} / 411357 = {100*tiny/411357:.2f}%")
print(f"kept 396899 / 425368 = {100*396899/425368:.2f}%  (cost {100-100*396899/425368:.1f}%)")
OUT["unclassified_pct"] = round(100 * unc / 425368, 2)
OUT["tiny_pct_of_labelled"] = round(100 * tiny / 411357, 2)
OUT["kept_pct"] = round(100 * 396899 / 425368, 1)

occ = pd.read_csv(R / "01_eda/occlusion_proxy_correlation.csv", index_col=0)
print("\nocclusion proxy Spearman rho:\n", occ.to_string())
cand = pd.read_csv(R / "01_eda/occlusion_candidates.csv")
flag = cand[cand.occlusion_flag]
print(f"\nocclusion sample {len(cand)} scenes; flagged {len(flag)}")
print(flag.groupby(["split", "disaster"]).size().to_string())

dup = pd.read_csv(R / "01_eda/near_duplicate_pairs_sample.csv")
cross = dup[dup.split_a != dup.split_b]
print(f"\nnear-duplicate pairs {len(dup)}, cross-split {len(cross)}")
top = pd.concat([dup.a, dup.b]).value_counts().head(4)
print("scenes appearing in most pairs:\n", top.to_string())
OUT["dup_pairs"] = len(dup)
OUT["dup_cross"] = len(cross)

# ------------------------------------------------------------------ patches
sec("3. PATCHES")
pb = pd.read_csv(R / "02_data_split/patch_class_balance.csv").set_index("split")
pb["total"] = pb.sum(axis=1)
print(pb.to_string())
print("grand total", int(pb["total"].sum()))
OUT["patches_total"] = int(pb["total"].sum())
share = 100 * pb.loc["train", ["no-damage", "minor-damage", "major-damage", "destroyed"]] / pb.loc["train", "total"]
print("\ntrain patch share %:\n", share.round(2).to_string())
share_ood = 100 * pb.loc["test_ood", ["no-damage", "minor-damage", "major-damage", "destroyed"]] / pb.loc["test_ood", "total"]
print("\ntest_ood patch share %:\n", share_ood.round(2).to_string())
OUT["train_patch_share"] = share.round(2).to_dict()
OUT["ood_patch_share"] = share_ood.round(2).to_dict()
print("\nclass weights:\n", pd.read_csv(R / "02_data_split/train_class_weights.csv").to_string(index=False))

# ------------------------------------------------------------------ models
sec("4. MODEL RESULTS (authoritative: results/model_comparison.csv)")
mc = pd.read_csv(R / "model_comparison.csv")
print(mc.to_string(index=False))

for tag, name in [("pretrained_siamese_resnet18", "P"), ("from_scratch_siamese_cnn", "S")]:
    d = mc[mc.model == tag].set_index("split")
    idr, oodr = d.loc["test_id"], d.loc["test_ood"]
    print(f"\n{tag}")
    for m in ["accuracy", "macro_F1", "macro_ROC_AUC"]:
        ab = idr[m] - oodr[m]
        print(f"  {m:14s} id {idr[m]:.4f} ood {oodr[m]:.4f} | abs drop {ab:.4f} rel {100*ab/idr[m]:.1f}%")
        OUT[f"{name}_{m}_reldrop"] = round(100 * ab / idr[m], 1)

pcP = pd.read_csv(R / "pretrained_resnet18/results_per_class.csv")
pcS = pd.read_csv(R / "scratch_cnn/results_per_class.csv")
print("\nPRETRAINED per-class:\n", pcP.round(3).to_string(index=False))
print("\nSCRATCH per-class:\n", pcS.round(3).to_string(index=False))

# derived OOD error structure -------------------------------------------------
sec("5. OOD ERROR STRUCTURE")
CL = ["no-damage", "minor-damage", "major-damage", "destroyed"]

p_ood = pcP[pcP.split == "test_ood"].set_index("class").loc[CL]
pred_counts_P = (p_ood.support * p_ood.recall / p_ood.precision).round(0)
print("PRETRAINED test_ood, derived from precision/recall/support:")
print("  true counts     ", p_ood.support.tolist())
print("  correct (TP)    ", (p_ood.support * p_ood.recall).round(0).tolist())
print("  predicted counts", pred_counts_P.tolist(), "sum", pred_counts_P.sum())
fa_P = 100 * (1 - p_ood.loc["no-damage", "recall"])
print(f"  false-alarm rate on true no-damage = {fa_P:.1f}%")
inter_pred_P = pred_counts_P.loc["minor-damage"] + pred_counts_P.loc["major-damage"]
inter_true = p_ood.loc["minor-damage", "support"] + p_ood.loc["major-damage", "support"]
print(f"  intermediate predicted {inter_pred_P:.0f} vs true {inter_true} = {inter_pred_P/inter_true:.1f}x over-prediction")
OUT["P_ood_false_alarm"] = round(float(fa_P), 1)
OUT["P_ood_inter_pred"] = int(inter_pred_P)
OUT["P_ood_inter_ratio"] = round(float(inter_pred_P / inter_true), 1)
OUT["P_ood_pred_counts"] = {c: int(v) for c, v in pred_counts_P.items()}

cm = pd.read_csv(R / "scratch_cnn/confusion_matrices.csv")
cm_ood = cm[cm.split == "test_ood"].set_index("true_class").loc[CL, CL].to_numpy()
print("\nSCRATCH test_ood confusion matrix (rows=true):\n", cm_ood)
off = cm_ood.sum() - np.trace(cm_ood)
over = np.triu(cm_ood, 1).sum()
under = np.tril(cm_ood, -1).sum()
print(f"  errors: {100*over/off:.1f}% over-estimate, {100*under/off:.1f}% under-estimate")
nd = cm_ood[0]
print(f"  false alarms on true no-damage: {nd[1:].sum()} / {nd.sum()} = {100*nd[1:].sum()/nd.sum():.1f}%")
pred_counts_S = cm_ood.sum(axis=0)
print("  predicted counts", pred_counts_S.tolist())
print(f"  intermediate predicted {pred_counts_S[1]+pred_counts_S[2]} vs true {inter_true} = "
      f"{(pred_counts_S[1]+pred_counts_S[2])/inter_true:.1f}x")
OUT["S_ood_over"] = round(float(100 * over / off), 1)
OUT["S_ood_under"] = round(float(100 * under / off), 1)
OUT["S_ood_false_alarm"] = round(float(100 * nd[1:].sum() / nd.sum()), 1)
OUT["S_ood_pred_counts"] = {c: int(v) for c, v in zip(CL, pred_counts_S)}
OUT["S_ood_inter_ratio"] = round(float((pred_counts_S[1] + pred_counts_S[2]) / inter_true), 1)

# training histories ----------------------------------------------------------
sec("6. TRAINING HISTORIES")
hp = pd.read_csv(R / "pretrained_resnet18/training_history.csv")
head = hp[hp.phase == "head"]
ft = hp[hp.phase == "finetune"]
print(f"pretrained: {len(hp)} epochs ({len(head)} head + {len(ft)} finetune)")
print(f"  best head macro-F1  {head.val_f1.max():.4f} (epoch {head.loc[head.val_f1.idxmax(),'epoch']})")
print(f"  first finetune epoch macro-F1 {ft.iloc[0].val_f1:.4f} (epoch {int(ft.iloc[0].epoch)})")
print(f"  best overall {hp.val_f1.max():.4f} at epoch {int(hp.loc[hp.val_f1.idxmax(),'epoch'])}")
print(f"  val loss min {hp.val_loss.min():.4f} at epoch {int(hp.loc[hp.val_loss.idxmin(),'epoch'])}, "
      f"final {hp.val_loss.iloc[-1]:.4f}")
best_ep = int(hp.loc[hp.val_f1.idxmax(), "epoch"])
print(f"  val loss at best-F1 epoch {hp.set_index('epoch').loc[best_ep,'val_loss']:.4f}")
lossmin_ep = int(hp.loc[hp.val_loss.idxmin(), "epoch"])
print(f"  macro-F1 at loss-min epoch {hp.set_index('epoch').loc[lossmin_ep,'val_f1']:.4f}")
OUT["P_head_best"] = round(float(head.val_f1.max()), 3)
OUT["P_ft_first"] = round(float(ft.iloc[0].val_f1), 3)
OUT["P_best_epoch"] = best_ep
OUT["P_epochs"] = len(hp)
OUT["P_lossmin_epoch"] = lossmin_ep
OUT["P_lossmin"] = round(float(hp.val_loss.min()), 3)
OUT["P_loss_at_best"] = round(float(hp.set_index("epoch").loc[best_ep, "val_loss"]), 3)
OUT["P_f1_at_lossmin"] = round(float(hp.set_index("epoch").loc[lossmin_ep, "val_f1"]), 3)

hs = pd.read_csv(R / "scratch_cnn/training_history.csv")
print(f"\nscratch: {len(hs)} epochs, best macro-F1 {hs.val_f1.max():.4f} at epoch "
      f"{int(hs.loc[hs.val_f1.idxmax(),'epoch'])}")
print(f"  val loss min {hs.val_loss.min():.4f} at epoch {int(hs.loc[hs.val_loss.idxmin(),'epoch'])}")
print(f"  wall clock {hs.seconds.sum()/60:.1f} min for {len(hs)} epochs "
      f"({hs.seconds.mean():.1f} s/epoch)")
OUT["S_epochs"] = len(hs)
OUT["S_best_epoch"] = int(hs.loc[hs.val_f1.idxmax(), "epoch"])
OUT["S_minutes"] = round(float(hs.seconds.sum() / 60), 1)
OUT["S_lossmin_epoch"] = int(hs.loc[hs.val_loss.idxmin(), "epoch"])

# ------------------------------------------------------------------ Part 3
sec("7. PART 3 SOCIOECONOMIC ANALYSIS")
t = pd.read_csv(R / "socio_analysis/tracts.csv")
b = pd.read_csv(R / "socio_analysis/buildings.csv", usecols=["damage_class", "event_name", "GEOID"])
print(f"buildings {len(b):,}, block groups {len(t)}, events {b.event_name.nunique()}")
print(b.event_name.value_counts().to_string())

SENT = -666666666
bad = (t.income_2017 <= 0) | (t.median_home_value_2017 <= 0) | (t.median_year_built_2017 < 1700)
print(f"\nCensus sentinel ({SENT}) contamination: {bad.sum()} of {len(t)} block groups")
print("  income", int((t.income_2017 <= 0).sum()),
      "| home value", int((t.median_home_value_2017 <= 0).sum()),
      "| year built", int((t.median_year_built_2017 < 1700).sum()))
print(f"pov_rate_2017 non-null: {t.pov_rate_2017.notna().sum()} (column is empty)")
print(f"exposure_proxy unique values: {t.exposure_proxy.unique()}  (hazard layer never populated)")
OUT["socio_buildings"] = len(b)
OUT["socio_units_raw"] = len(t)
OUT["socio_bad_units"] = int(bad.sum())

clean = t[~bad].copy()
print(f"\nclean analysis sample: {len(clean)} block groups")
OUT["socio_units_clean"] = len(clean)
clean["building_age"] = 2017 - clean.median_year_built_2017


def ols(y, X, names):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(np.abs(tstat), n - k))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({"feature": ["intercept"] + names, "coef": beta, "se": se,
                         "t": tstat, "p": p}), r2, n


def z(col):
    return (col - col.mean()) / col.std(ddof=0)


y = clean.damage_score_mean.to_numpy()
socio = ["income_2017", "median_home_value_2017", "renter_pct_2017"]
expo = ["building_density", "building_age"]
Xn = np.column_stack([z(clean[c]) for c in socio])
Xc = np.column_stack([z(clean[c]) for c in socio + expo])

naive, r2n, n = ols(y, Xn, socio)
ctrl, r2c, _ = ols(y, Xc, socio + expo)
print(f"\nNAIVE  (n={n}, R2={r2n:.4f})\n", naive.round(4).to_string(index=False))
print(f"\nCONTROLLED (n={n}, R2={r2c:.4f})\n", ctrl.round(4).to_string(index=False))
OUT["socio_r2_naive"] = round(float(r2n), 4)
OUT["socio_r2_ctrl"] = round(float(r2c), 4)
OUT["socio_naive"] = naive.round(4).to_dict("records")
OUT["socio_ctrl"] = ctrl.round(4).to_dict("records")

# quintile view: much easier to read in a report than a standardised beta
clean["income_q"] = pd.qcut(clean.income_2017, 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
q = clean.groupby("income_q", observed=True).agg(
    block_groups=("GEOID", "size"), buildings=("building_count", "sum"),
    median_income=("income_2017", "median"), damage_score=("damage_score_mean", "mean"),
    share_destroyed=("share_destroyed", "mean"), share_none=("share_no-damage", "mean"))
print("\nDamage by income quintile:\n", q.round(3).to_string())
OUT["socio_quintiles"] = q.round(4).reset_index().to_dict("records")

clean.to_csv(Path(__file__).with_name("tracts_clean.csv"), index=False)
pd.concat([naive.assign(model="naive"), ctrl.assign(model="controlled")]).to_csv(
    Path(__file__).with_name("socio_regression_clean.csv"), index=False)

with open(Path(__file__).with_name("verified_numbers.json"), "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\n\nwrote verified_numbers.json / socio_regression_clean.csv / tracts_clean.csv")
