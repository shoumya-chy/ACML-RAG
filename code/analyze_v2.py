"""v2 analysis.
1. Dose-response curves (graded irrelevance severity; contra position; compound noise)
2. Silent-failure metric per noise type
3. Risk estimator AUROC: retrieval / model / delta (confidence-gain) / combined
4. Conformal selective generation under noise shift: 5 methods
   - marginal CRC
   - heuristic-group CRC
   - learned-group CRC (k-means on noise descriptors, per-group threshold)
   - shift-weighted CRC (density-ratio weighted calibration; Tibshirani-style)
   - worst-group CRC (conservative bound)
5. Cross-model risk-estimator transfer (train on model A, test on model B)
Usage: python3 analyze_v2.py [tag1] [tag2]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(11)

TAG1 = sys.argv[1] if len(sys.argv) > 1 else "flan_t5_base"
TAG2 = sys.argv[2] if len(sys.argv) > 2 else None

RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "ret_std", "conflict", "n_claims", "has_ctx"]
MOD = ["lp_mean", "lp_min", "ent_mean", "ent_max", "ent_first", "ans_len"]
DELTA = ["d_lp", "d_ent", "ans_changed"]
ALL = RET + MOD + DELTA
CONDS = ["clean", "irr25", "irr50", "irr75", "irr100",
         "contra_r1", "contra_r4", "contra_only", "distractor", "mixed", "closedbook"]

def load(tag):
    f = f"{DATA}/results_{tag}.json"
    if not os.path.exists(f):
        f = f"{DATA}/results_{tag}_partial.json"
    return json.load(open(f))

def arr(rs, keys): return np.array([[r[k] for k in keys] for r in rs])
def lab(rs): return np.array([1 if r["label"] == "hallucination" else 0 for r in rs])

rows = load(TAG1)
print(f"== {TAG1}: {len(rows)} rows ==")
report = {"model": TAG1, "n_rows": len(rows)}

# ---------- 1. dose-response ----------
print("\n=== Dose-response (hallucination | abstain | correct) ===")
dr = {}
for cond in CONDS:
    for regime in ["synthetic", "real"]:
        sub = [r for r in rows if r["cond"] == cond and r["regime"] == regime]
        if not sub: continue
        n = len(sub)
        h = sum(r["label"] == "hallucination" for r in sub) / n
        a = sum(r["label"] == "abstain" for r in sub) / n
        dr[(cond, regime)] = (h, a, 1 - h - a, n)
        print(f"{cond:11s} {regime:9s} n={n:3d}  H={h:.2f}  A={a:.2f}  C={1-h-a:.2f}")
report["dose_response"] = {f"{c}|{g}": v for (c, g), v in dr.items()}

# severity curve figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sev = ["clean", "irr25", "irr50", "irr75", "irr100"]
for regime, col in [("synthetic", "#4C72B0"), ("real", "#DD8452")]:
    hh = [dr[(c, regime)][0] for c in sev if (c, regime) in dr]
    aa = [dr[(c, regime)][1] for c in sev if (c, regime) in dr]
    ax[0].plot(range(len(hh)), hh, "o-", color=col, label=f"halluc ({regime})")
    ax[0].plot(range(len(aa)), aa, "s--", color=col, alpha=0.5, label=f"abstain ({regime})")
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax[0].set_xlabel("irrelevance noise severity"); ax[0].set_ylabel("rate")
ax[0].set_title("Graded irrelevance dose-response"); ax[0].legend(fontsize=7)
cnds = ["contra_r1", "contra_r4", "contra_only", "distractor", "mixed"]
x = np.arange(len(cnds)); w = 0.38
for off, regime, col in [(-w/2, "synthetic", "#4C72B0"), (w/2, "real", "#DD8452")]:
    vals = [dr[(c, regime)][0] if (c, regime) in dr else 0 for c in cnds]
    ax[1].bar(x + off, vals, w, label=regime, color=col)
ax[1].set_xticks(x); ax[1].set_xticklabels(cnds, rotation=20)
ax[1].set_ylabel("hallucination rate"); ax[1].set_title("Conflict-type & position effects")
ax[1].legend(); fig.tight_layout(); fig.savefig(f"{FIG}/v2_dose_response.png", dpi=140)

# ---------- 2. silent failure ----------
print("\n=== Silent-failure rate (share of noise-affected errors with NO abstention signal) ===")
sf = {}
for cond in CONDS:
    sub = [r for r in rows if r["cond"] == cond]
    if not sub: continue
    errs = [r for r in sub if r["label"] != "correct"]
    if not errs: sf[cond] = 0.0; continue
    sf[cond] = sum(r["label"] == "hallucination" for r in errs) / len(errs)
    print(f"{cond:11s} silent-failure={sf[cond]:.2f}  (n_err={len(errs)})")
report["silent_failure"] = sf

# ---------- 3. risk estimator ----------
answered = [r for r in rows if r["label"] != "abstain"]
groups = np.array([r["qid"] for r in answered])
y = lab(answered)
print(f"\n=== Risk estimator AUROC (answered n={len(y)}, halluc rate={y.mean():.3f}) ===")

def oof_probs(keys, name=""):
    cache = f"/tmp/pilot/oof_{TAG1}_{name}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    X = arr(answered, keys)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    np.save(cache, p)
    return p

probs = {}
for name, keys in [("retrieval", RET), ("model", MOD), ("delta", DELTA),
                   ("model+delta", MOD + DELTA), ("combined", ALL)]:
    probs[name] = oof_probs(keys, name)
    auc = roc_auc_score(y, probs[name])
    report[f"auc_{name}"] = float(auc)
    print(f"{name:12s} AUROC={auc:.3f}")

qids = sorted(set(groups))
def boot_gap(pa, pb, n=2000):
    g = []
    for _ in range(n):
        bq = set(rng.choice(qids, len(qids), replace=True))
        m = np.array([gg in bq for gg in groups])
        if len(set(y[m])) < 2: continue
        g.append(roc_auc_score(y[m], pa[m]) - roc_auc_score(y[m], pb[m]))
    return float(np.mean(g)), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))
gap = boot_gap(probs["combined"], probs["model"])
report["gap_combined_vs_model"] = gap
print(f"combined - model: {gap[0]:+.3f} [95% CI {gap[1]:+.3f}, {gap[2]:+.3f}]")
gap2 = boot_gap(probs["model+delta"], probs["model"])
report["gap_delta_vs_model"] = gap2
print(f"model+delta - model: {gap2[0]:+.3f} [95% CI {gap2[1]:+.3f}, {gap2[2]:+.3f}]")

# ---------- 4. conformal under noise shift: 5 methods ----------
print("\n=== Conformal selective generation under noise shift (alpha=0.10, NREPS resplits) ===")
ALPHA = 0.10
p_all = probs["combined"]
noisy = {"irr75", "irr100", "contra_r1", "contra_r4", "contra_only", "mixed"}
is_noisy = np.array([r["cond"] in noisy for r in answered])
Xnoise = arr(answered, RET + DELTA)            # observable noise descriptors

def crc_threshold(scores, labels, alpha, weights=None):
    if weights is None: weights = np.ones(len(scores))
    order = np.argsort(scores)
    s, l, w = scores[order], labels[order], weights[order]
    cw = np.cumsum(w); cl = np.cumsum(w * l)
    ok = (cl + 1) / (cw + 1) <= alpha
    if not ok.any(): return -np.inf
    return s[np.where(ok)[0].max()]

def eval_rule(scores, labels, t):
    m = scores <= t
    if m.sum() == 0: return 0.0, 0.0
    return float(labels[m].mean()), float(m.mean())

from scipy.stats import beta as beta_dist
def cp_upper(k, n, delta=0.10):
    """Clopper-Pearson upper confidence bound on a binomial proportion."""
    if n == 0: return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k)) if k < n else 1.0

def riskbin_rule(pc, yc, pt, alpha, nbins=10):
    """Risk-binned Mondrian CRC: certify each calibration-score bin with a CP upper
    bound; answer test points whose score falls in certified bins. Shift-robust when
    the score sigma-algebra captures the covariate shift."""
    qs = np.quantile(pc, np.linspace(0, 1, nbins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    certified = []
    for b in range(nbins):
        m = (pc > qs[b]) & (pc <= qs[b + 1])
        if m.sum() == 0: continue
        if cp_upper(int(yc[m].sum()), int(m.sum())) <= alpha:
            certified.append(b)
    tb = np.digitize(pt, qs) - 1
    acc = np.isin(tb, certified)
    return acc

res = {k: [] for k in ["marginal", "heuristic", "learned", "weighted", "worstgroup", "riskbin"]}
conf = arr(answered, ["conflict"]).ravel()
rmean = arr(answered, ["ret_mean"]).ravel()
hasctx = arr(answered, ["has_ctx"]).ravel()
med = np.median(rmean[hasctx == 1])
heur_grp = ((conf == 1) | (rmean < med * 0.6) | (hasctx == 0)).astype(int)

import pickle
RESCK = f"/tmp/pilot/res_{TAG1}.pkl"
if os.path.exists(RESCK):
    res = pickle.load(open(RESCK, "rb"))
start_rep = max((len(v) for v in res.values()), default=0)
end_rep = int(os.environ.get("NREPS", "200"))
for rep in range(start_rep, end_rep):
    rr = np.random.default_rng(5000 + rep)
    qsh = rr.permutation(qids)
    calq, teq = set(qsh[:len(qids)//2]), set(qsh[len(qids)//2:])
    cal = np.array([g in calq for g in groups])
    te = np.array([g in teq for g in groups])
    cal_keep = cal & (~is_noisy | (rr.random(len(cal)) < 0.30))
    te_keep = te & (is_noisy | (rr.random(len(te)) < 0.50))
    if cal_keep.sum() < 50 or te_keep.sum() < 50: continue

    # 1 marginal
    t = crc_threshold(p_all[cal_keep], y[cal_keep], ALPHA)
    res["marginal"].append(eval_rule(p_all[te_keep], y[te_keep], t))

    # 2 heuristic groups
    rk, ck, nk = [], [], []
    for g in [0, 1]:
        cm = cal_keep & (heur_grp == g); tm = te_keep & (heur_grp == g)
        if cm.sum() < 10 or tm.sum() == 0: continue
        tg = crc_threshold(p_all[cm], y[cm], ALPHA)
        r, c = eval_rule(p_all[tm], y[tm], tg)
        rk.append(r * tm.sum()); ck.append(c * tm.sum()); nk.append(tm.sum())
    if nk: res["heuristic"].append((sum(rk)/sum(nk), sum(ck)/sum(nk)))

    # 3 learned groups: k-means on noise descriptors (fit on cal+test pool, unsupervised)
    sc = StandardScaler().fit(Xnoise[cal_keep | te_keep])
    km = KMeans(n_clusters=4, n_init=3, random_state=rep).fit(sc.transform(Xnoise[cal_keep]))
    gcal = km.predict(sc.transform(Xnoise[cal_keep])); gte = km.predict(sc.transform(Xnoise[te_keep]))
    pc, yc = p_all[cal_keep], y[cal_keep]; pt, yt = p_all[te_keep], y[te_keep]
    rk, ck, nk = [], [], []
    for g in range(4):
        cm, tm = gcal == g, gte == g
        if tm.sum() == 0: continue
        if cm.sum() < 15:
            tg = crc_threshold(pc, yc, ALPHA)   # fallback marginal
        else:
            tg = crc_threshold(pc[cm], yc[cm], ALPHA)
        r, c = eval_rule(pt[tm], yt[tm], tg)
        rk.append(r * tm.sum()); ck.append(c * tm.sum()); nk.append(tm.sum())
    if nk: res["learned"].append((sum(rk)/sum(nk), sum(ck)/sum(nk)))

    # 4 shift-weighted CRC: density ratio w(x)=P(test|x)/P(cal|x) via logistic regression
    Z = np.vstack([Xnoise[cal_keep], Xnoise[te_keep]])
    zz = np.concatenate([np.zeros(cal_keep.sum()), np.ones(te_keep.sum())])
    sc2 = StandardScaler().fit(Z)
    lr = LogisticRegression(max_iter=1000).fit(sc2.transform(Z), zz)
    pr = lr.predict_proba(sc2.transform(Xnoise[cal_keep]))[:, 1]
    wgt = np.clip(pr / (1 - pr + 1e-9), 0.05, 20.0)
    t = crc_threshold(p_all[cal_keep], y[cal_keep], ALPHA, weights=wgt)
    res["weighted"].append(eval_rule(p_all[te_keep], y[te_keep], t))

    # 6 risk-binned Mondrian CRC
    acc = riskbin_rule(p_all[cal_keep], y[cal_keep], p_all[te_keep], ALPHA)
    if acc.sum() > 0:
        res["riskbin"].append((float(y[te_keep][acc].mean()), float(acc.mean())))
    else:
        res["riskbin"].append((0.0, 0.0))

    # 5 worst-group (conservative): min threshold over learned groups
    ts = []
    for g in range(4):
        cm = gcal == g
        if cm.sum() >= 15: ts.append(crc_threshold(pc[cm], yc[cm], ALPHA))
    if ts:
        res["worstgroup"].append(eval_rule(pt, yt, min(ts)))

pickle.dump(res, open(RESCK, 'wb'))
print(f"{'method':11s} {'risk':>6s} {'viol%':>6s} {'answer%':>8s}")
for k in res:
    a = np.array(res[k])
    if len(a) == 0: continue
    viol = float((a[:, 0] > ALPHA).mean())
    report[f"crc_{k}"] = dict(risk=float(a[:,0].mean()), viol=viol, coverage=float(a[:,1].mean()))
    print(f"{k:11s} {a[:,0].mean():6.3f} {viol*100:6.0f} {a[:,1].mean()*100:8.1f}")

# violin/hist figure
fig, ax = plt.subplots(figsize=(7.5, 4.2))
names = [k for k in res if len(res[k])]
data = [np.array(res[k])[:, 0] for k in names]
parts = ax.violinplot(data, showmeans=True)
ax.axhline(ALPHA, color="k", ls=":", label="target alpha=0.10")
ax.set_xticks(range(1, len(names)+1)); ax.set_xticklabels(names)
ax.set_ylabel("realized hallucination risk among answered")
ax.set_title(f"Guarantee validity under retrieval-noise shift ({TAG1})")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/v2_crc_methods.png", dpi=140)

# ---------- 5. cross-model transfer ----------
if TAG2:
    rows2 = load(TAG2)
    answered2 = [r for r in rows2 if r["label"] != "abstain"]
    y2 = lab(answered2)
    print(f"\n=== Cross-model transfer: train {TAG2} -> test {TAG1} (and reverse) ===")
    for (ra, ya, rb, yb, nm) in [(answered2, y2, answered, y, f"{TAG2}->{TAG1}"),
                                 (answered, y, answered2, y2, f"{TAG1}->{TAG2}")]:
        m = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3)
        m.fit(arr(ra, ALL), ya)
        p = m.predict_proba(arr(rb, ALL))[:, 1]
        auc = roc_auc_score(yb, p)
        # calibration transfer: threshold from source cal, realized risk on target
        t = crc_threshold(m.predict_proba(arr(ra, ALL))[:, 1], ya, ALPHA)
        r, c = eval_rule(p, yb, t)
        report[f"transfer_{nm}"] = dict(auc=float(auc), risk=float(r), coverage=float(c))
        print(f"{nm:28s} AUROC={auc:.3f}  transferred-threshold risk={r:.3f} (target {ALPHA}) answer={c:.2f}")

json.dump(report, open(f"{DATA}/analysis_v2_{TAG1}.json", "w"), indent=1)
print("\nsaved figs + analysis json")
