"""Sensitivity analyses for risk-binned Mondrian CRC vs marginal CRC.
(a) target alpha sweep; (b) bin-count sweep; (c) shift-severity sweep.
Uses cached out-of-fold risk scores if present, else recomputes.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figs")
TAG = sys.argv[1] if len(sys.argv) > 1 else "flan_t5_base"
NREPS = int(os.environ.get("NREPS", "60"))

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
    p_all = np.load(cache)
else:
    p_all = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3)
        m.fit(X[tr], y[tr]); p_all[te] = m.predict_proba(X[te])[:, 1]
    os.makedirs("/tmp/pilot", exist_ok=True)
    np.save(cache, p_all)

noisy = {"irr75", "irr100", "contra_r1", "contra_r4", "contra_only", "mixed"}
is_noisy = np.array([r["cond"] in noisy for r in answered])
qids = sorted(set(groups))

def cp_upper(k, n, delta=0.10):
    if n == 0: return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k)) if k < n else 1.0

def riskbin(pc, yc, pt, yt, alpha, nbins=10):
    qs = np.quantile(pc, np.linspace(0, 1, nbins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cert = [b for b in range(nbins)
            if ((pc > qs[b]) & (pc <= qs[b+1])).sum() > 0
            and cp_upper(int(yc[(pc > qs[b]) & (pc <= qs[b+1])].sum()),
                         int(((pc > qs[b]) & (pc <= qs[b+1])).sum())) <= alpha]
    tb = np.digitize(pt, qs) - 1
    acc = np.isin(tb, cert)
    if acc.sum() == 0: return 0.0, 0.0
    return float(yt[acc].mean()), float(acc.mean())

def marginal(pc, yc, pt, yt, alpha):
    order = np.argsort(pc); s, l = pc[order], yc[order]
    cl = np.cumsum(l); cn = np.arange(1, len(s) + 1)
    ok = (cl + 1) / (cn + 1) <= alpha
    if not ok.any(): return 0.0, 0.0
    t = s[np.where(ok)[0].max()]
    m = pt <= t
    if m.sum() == 0: return 0.0, 0.0
    return float(yt[m].mean()), float(m.mean())

def splits(rep, cal_noisy_keep=0.30, te_clean_keep=0.50):
    rr = np.random.default_rng(7000 + rep)
    qsh = rr.permutation(qids)
    calq, teq = set(qsh[:len(qids)//2]), set(qsh[len(qids)//2:])
    cal = np.array([g in calq for g in groups])
    te = np.array([g in teq for g in groups])
    cal_keep = cal & (~is_noisy | (rr.random(len(cal)) < cal_noisy_keep))
    te_keep = te & (is_noisy | (rr.random(len(te)) < te_clean_keep))
    return cal_keep, te_keep

out = {"alpha_sweep": {}, "bin_sweep": {}, "shift_sweep": {}}

for alpha in [0.05, 0.10, 0.15, 0.20]:
    accm, accr = [], []
    for rep in range(NREPS):
        ck, tk = splits(rep)
        accm.append(marginal(p_all[ck], y[ck], p_all[tk], y[tk], alpha))
        accr.append(riskbin(p_all[ck], y[ck], p_all[tk], y[tk], alpha))
    am, ar = np.array(accm), np.array(accr)
    out["alpha_sweep"][alpha] = dict(
        marg_risk=float(am[:,0].mean()), marg_viol=float((am[:,0] > alpha).mean()), marg_cov=float(am[:,1].mean()),
        rb_risk=float(ar[:,0].mean()), rb_viol=float((ar[:,0] > alpha).mean()), rb_cov=float(ar[:,1].mean()))
    print(f"alpha={alpha:.2f}  marginal risk={am[:,0].mean():.3f} viol={(am[:,0]>alpha).mean():.2f} cov={am[:,1].mean():.2f} | "
          f"riskbin risk={ar[:,0].mean():.3f} viol={(ar[:,0]>alpha).mean():.2f} cov={ar[:,1].mean():.2f}", flush=True)

for B in [5, 10, 20, 40]:
    acc = []
    for rep in range(NREPS):
        ck, tk = splits(rep)
        acc.append(riskbin(p_all[ck], y[ck], p_all[tk], y[tk], 0.10, nbins=B))
    a = np.array(acc)
    out["bin_sweep"][B] = dict(risk=float(a[:,0].mean()), viol=float((a[:,0] > 0.10).mean()), cov=float(a[:,1].mean()))
    print(f"B={B:2d}  riskbin risk={a[:,0].mean():.3f} viol={(a[:,0]>0.10).mean():.2f} cov={a[:,1].mean():.2f}", flush=True)

for f in [0.9, 0.7, 0.5, 0.3, 0.1]:
    accm, accr = [], []
    for rep in range(NREPS):
        ck, tk = splits(rep, cal_noisy_keep=f)
        accm.append(marginal(p_all[ck], y[ck], p_all[tk], y[tk], 0.10))
        accr.append(riskbin(p_all[ck], y[ck], p_all[tk], y[tk], 0.10))
    am, ar = np.array(accm), np.array(accr)
    out["shift_sweep"][f] = dict(
        marg_risk=float(am[:,0].mean()), marg_viol=float((am[:,0] > 0.10).mean()),
        rb_risk=float(ar[:,0].mean()), rb_viol=float((ar[:,0] > 0.10).mean()), rb_cov=float(ar[:,1].mean()))
    print(f"cal_noisy_keep={f:.1f}  marginal risk={am[:,0].mean():.3f} viol={(am[:,0]>0.10).mean():.2f} | "
          f"riskbin risk={ar[:,0].mean():.3f} viol={(ar[:,0]>0.10).mean():.2f} cov={ar[:,1].mean():.2f}", flush=True)

json.dump(out, open(f"{DATA}/sensitivity_{TAG}.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
als = sorted(out["alpha_sweep"])
ax[0].plot(als, [out["alpha_sweep"][a]["marg_risk"] for a in als], "o-", label="marginal", color="#C44E52")
ax[0].plot(als, [out["alpha_sweep"][a]["rb_risk"] for a in als], "s-", label="risk-binned", color="#55A868")
ax[0].plot(als, als, "k:", label="target")
ax[0].set_xlabel(r"target $\alpha$"); ax[0].set_ylabel("realized risk"); ax[0].set_title("Target-level sweep"); ax[0].legend(fontsize=8)
bs = sorted(out["bin_sweep"])
axb = ax[1]
axb.plot(bs, [out["bin_sweep"][b]["risk"] for b in bs], "s-", color="#55A868", label="risk")
axb.axhline(0.10, color="k", ls=":")
axb2 = axb.twinx()
axb2.plot(bs, [out["bin_sweep"][b]["cov"] for b in bs], "^--", color="#4C72B0", label="answer rate")
axb.set_xlabel("number of bins $B$"); axb.set_ylabel("realized risk"); axb2.set_ylabel("answer rate")
axb.set_title(r"Bin-count sweep ($\alpha=0.10$)")
fs = sorted(out["shift_sweep"], reverse=True)
sev = [1 - f for f in fs]
ax[2].plot(sev, [out["shift_sweep"][f]["marg_risk"] for f in fs], "o-", color="#C44E52", label="marginal")
ax[2].plot(sev, [out["shift_sweep"][f]["rb_risk"] for f in fs], "s-", color="#55A868", label="risk-binned")
ax[2].axhline(0.10, color="k", ls=":", label="target")
ax[2].set_xlabel("shift severity (1 - calibration noise retention)"); ax[2].set_ylabel("realized risk")
ax[2].set_title("Shift-severity sweep"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/v2_sensitivity.png", dpi=140)
print("saved sensitivity json + figure")
