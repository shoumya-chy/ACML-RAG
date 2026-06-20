"""Estimator diagnostics: per-condition and per-regime AUROC, reliability diagram,
feature importances. Pure analysis on logged data; no new generations.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figs")
TAG = sys.argv[1] if len(sys.argv) > 1 else "flan_t5_base"

rows = json.load(open(f"{DATA}/results_{TAG}.json"))
RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "ret_std", "conflict", "n_claims", "has_ctx"]
MOD = ["lp_mean", "lp_min", "ent_mean", "ent_max", "ent_first", "ans_len"]
DELTA = ["d_lp", "d_ent", "ans_changed"]
ALL = RET + MOD + DELTA
answered = [r for r in rows if r["label"] != "abstain"]
groups = np.array([r["qid"] for r in answered])
y = np.array([1 if r["label"] == "hallucination" else 0 for r in answered])
X = np.array([[r[k] for k in ALL] for r in answered])

cache = f"/tmp/pilot/oof_{TAG}_combined.npy"
if os.path.exists(cache):
    p = np.load(cache)
else:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    np.save(cache, p)

out = {}
print("=== Per-condition AUROC (combined estimator, out-of-fold) ===")
conds = np.array([r["cond"] for r in answered])
for c in sorted(set(conds)):
    m = conds == c
    if len(set(y[m])) < 2:
        print(f"{c:11s} n={m.sum():4d}  (single class, base rate {y[m].mean():.2f})")
        out[f"auc_{c}"] = None
        continue
    a = roc_auc_score(y[m], p[m])
    out[f"auc_{c}"] = float(a)
    print(f"{c:11s} n={m.sum():4d}  AUROC={a:.3f}  halluc rate={y[m].mean():.2f}")

print("\n=== Per-regime AUROC ===")
regs = np.array([r["regime"] for r in answered])
for g in ["synthetic", "real"]:
    m = regs == g
    a = roc_auc_score(y[m], p[m])
    out[f"auc_{g}"] = float(a)
    print(f"{g:9s} n={m.sum():4d}  AUROC={a:.3f}")

# Expected calibration error of the raw estimator (10 bins)
frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
ece = float(np.mean(np.abs(frac - mean_p)))
out["ece"] = ece
print(f"\nECE (10 quantile bins): {ece:.3f}")

# Feature importances (full-data fit, for interpretation only)
m = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3).fit(X, y)
imp = sorted(zip(ALL, m.feature_importances_), key=lambda t: -t[1])
out["importances"] = {k: float(v) for k, v in imp}
print("\n=== Feature importances (top 10) ===")
for k, v in imp[:10]:
    print(f"{k:12s} {v:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(mean_p, frac, "o-", color="#4C72B0", label="estimator")
ax[0].plot([0, 1], [0, 1], "k:", label="perfect")
ax[0].set_xlabel("predicted hallucination probability")
ax[0].set_ylabel("observed frequency")
ax[0].set_title(f"Reliability ({TAG}, ECE={ece:.3f})"); ax[0].legend()
ks = [k for k, _ in imp][:10][::-1]
vs = [v for _, v in imp][:10][::-1]
cols = ["#55A868" if k in RET else "#DD8452" if k in DELTA else "#4C72B0" for k in ks]
ax[1].barh(ks, vs, color=cols)
ax[1].set_title("Feature importance (green=retrieval, orange=contrast, blue=model)")
fig.tight_layout(); fig.savefig(f"{FIG}/v2_diagnostics.png", dpi=140)
json.dump(out, open(f"{DATA}/diagnostics_{TAG}.json", "w"), indent=1)
print("\nsaved diagnostics json + figure")
