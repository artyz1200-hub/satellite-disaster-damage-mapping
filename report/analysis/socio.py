"""Part 3, redone on the cleaned block groups, with event fixed effects."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# <repo>/report/analysis/<this file>  ->  <repo>
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent

t = pd.read_csv(HERE / "tracts_clean.csv")
b = pd.read_csv(ROOT / "results/socio_analysis/buildings.csv",
                usecols=["GEOID", "event_name", "damage_class"])

# dominant event per block group (they are essentially single-event by construction)
ev = b.groupby("GEOID").event_name.agg(lambda s: s.value_counts().idxmax())
purity = b.groupby("GEOID").event_name.agg(lambda s: s.value_counts().iloc[0] / len(s))
t = t.merge(ev.rename("event"), left_on="GEOID", right_index=True, how="left")
t = t.merge(purity.rename("purity"), left_on="GEOID", right_index=True, how="left")
print("block groups:", len(t))
print("single-event block groups (purity == 1):", int((t.purity == 1).sum()))
print("\nevent mix of the analysis sample:")
print(t.groupby("event").agg(block_groups=("GEOID", "size"), buildings=("building_count", "sum"),
                             median_income=("income_2017", "median"),
                             damage=("damage_score_mean", "mean")).round(3).to_string())

print("\npredictor correlations:")
cols = ["damage_score_mean", "income_2017", "median_home_value_2017", "renter_pct_2017",
        "building_density", "building_age"]
print(t[cols].corr().round(3).to_string())


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


def ols(y, X, names, label):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    n, k = X.shape
    cov = (r @ r / (n - k)) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tt = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(tt), n - k))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    out = pd.DataFrame({"model": label, "feature": ["intercept"] + names,
                        "coef": beta, "se": se, "t": tt, "p": p})
    print(f"\n--- {label}  n={n}  R2={r2:.4f}  adjR2={adj:.4f}")
    print(out.round(4).to_string(index=False))
    return out, r2, adj


y = t.damage_score_mean.to_numpy()
socio = ["income_2017", "median_home_value_2017", "renter_pct_2017"]
built = ["building_density", "building_age"]

rows, fits = [], {}
m, r2, adj = ols(y, z(t.income_2017).to_numpy()[:, None], ["income_2017"], "income only")
rows.append(m); fits["income only"] = (r2, adj)
m, r2, adj = ols(y, np.column_stack([z(t[c]) for c in socio]), socio, "naive (socioeconomic)")
rows.append(m); fits["naive (socioeconomic)"] = (r2, adj)
m, r2, adj = ols(y, np.column_stack([z(t[c]) for c in socio + built]), socio + built,
                 "controlled (+ built environment)")
rows.append(m); fits["controlled (+ built environment)"] = (r2, adj)

dummies = pd.get_dummies(t.event, drop_first=True).astype(float)
Xfe = np.column_stack([z(t[c]) for c in socio + built] + [dummies.to_numpy()])
m, r2, adj = ols(y, Xfe, socio + built + list(dummies.columns), "+ event fixed effects")
rows.append(m); fits["+ event fixed effects"] = (r2, adj)

print("\n\nMODEL FIT LADDER")
for k, (r2, adj) in fits.items():
    print(f"  {k:34s} R2 {r2:.4f}  adjR2 {adj:.4f}")

print("\n\nDAMAGE BY INCOME QUINTILE (whole sample)")
t["q"] = pd.qcut(t.income_2017, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
q = t.groupby("q", observed=True).agg(
    groups=("GEOID", "size"), buildings=("building_count", "sum"),
    med_income=("income_2017", "median"), damage=("damage_score_mean", "mean"),
    destroyed=("share_destroyed", "mean"), none=("share_no-damage", "mean"))
print(q.round(3).to_string())

print("\n\nDAMAGE BY INCOME TERCILE COMPUTED WITHIN EACH EVENT")
big = t[t.event.map(t.event.value_counts()) >= 15].copy()
print(f"events with at least 15 block groups: {big.event.nunique()} ({len(big)} groups)")
big["q_within"] = big.groupby("event").income_2017.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
qw = big.groupby("q_within", observed=True).agg(
    groups=("GEOID", "size"), damage=("damage_score_mean", "mean"),
    destroyed=("share_destroyed", "mean"))
print(qw.round(3).to_string())
per_ev = big.pivot_table(index="event", columns="q_within", values="damage_score_mean",
                         aggfunc="mean", observed=True)
print("\nmean damage score by within-event income tercile:")
print(per_ev.round(3).to_string())
qw.to_csv(HERE / "socio_within_event.csv")

pd.concat(rows).to_csv(HERE / "socio_regression_final.csv", index=False)
q.to_csv(HERE / "socio_quintiles.csv")
t.to_csv(HERE / "tracts_final.csv", index=False)
pd.DataFrame([{"model": k, "R2": v[0], "adjR2": v[1]} for k, v in fits.items()]).to_csv(
    HERE / "socio_fits.csv", index=False)
print("\nwrote socio_regression_final.csv / socio_quintiles.csv / socio_fits.csv / tracts_final.csv")
