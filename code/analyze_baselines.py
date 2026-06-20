"""Compare real UQ baselines (semantic entropy, P(True)) against our
noise-conditional estimator, on (a) hallucination prediction AUROC and
(b) selective-generation validity under retrieval-noise shift.
Reads data/results_se_flan_t5_small.json.
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
rows = json.load(open(f"{DATA}/results_se_flan_t5_small.json"))
ans = [r for r in rows if r["label"] != "abstain"]
y = np.array([1 if r["label"] == "hallucination" else 0 for r in ans])
groups = np.array([r["qid"] for r in ans])
qids = sorted(set(groups))
print(f"records={len(rows)} answered={len(y)} halluc_rate={y.mean():.3f}")

def col(k): return np.array([r[k] for r in ans], float)
se = col("se"); ptrue = col("p_true")
RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "has_ctx"]
Xret = np.array([[r[k] for k in RET] for r in ans])

# --- single-signal baseline AUROCs (higher score = more risk) ---
# semantic entropy: high entropy -> risk. P(True): low p_true -> risk, so use 1-p_true.
auc = {}
auc["semantic_entropy"] = roc_auc_score(y, se)
auc["one_minus_ptrue"] = roc_auc_score(y, 1 - ptrue)
# our noise-conditional estimator (retrieval descriptors), grouped CV
def oof(X):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=3)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
p_ours = oof(Xret)
auc["ours_retrieval"] = roc_auc_score(y, p_ours)
# ours + SE + ptrue combined
Xall = np.column_stack([Xret, se, ptrue])
p_comb = oof(Xall)
auc["ours_plus_se_ptrue"] = roc_auc_score(y, p_comb)
print("\n=== Hallucination-prediction AUROC ===")
for k in ["semantic_entropy", "one_minus_ptrue", "ours_retrieval", "ours_plus_se_ptrue"]:
    print(f"{k:22s} {auc[k]:.3f}")

# bootstrap: does SE add to ours?
rng = np.random.default_rng(0)
gaps = []
for _ in range(2000):
    bq = set(rng.choice(qids, len(qids), replace=True))
    m = np.array([g in bq for g in groups])
    if len(set(y[m])) < 2: continue
    gaps.append(roc_auc_score(y[m], p_comb[m]) - roc_auc_score(y[m], p_ours[m]))
print(f"(ours+SE+pTrue) - ours: {np.mean(gaps):+.3f} [95% CI {np.percentile(gaps,2.5):+.3f}, {np.percentile(gaps,97.5):+.3f}]")
gaps2 = []
for _ in range(2000):
    bq = set(rng.choice(qids, len(qids), replace=True))
    m = np.array([g in bq for g in groups])
    if len(set(y[m])) < 2: continue
    gaps2.append(roc_auc_score(y[m], p_ours[m]) - roc_auc_score(y[m], se[m]))
print(f"ours - semantic_entropy: {np.mean(gaps2):+.3f} [95% CI {np.percentile(gaps2,2.5):+.3f}, {np.percentile(gaps2,97.5):+.3f}]")

# --- selective generation under shift: SE-threshold vs risk-binned(ours) ---
ALPHA = 0.10
noisy = {"irr100", "contra_r1", "contra_only"}
is_noisy = np.array([r["cond"] in noisy for r in ans])
def cp_upper(k, n, d=0.10):
    if n == 0: return 1.0
    return float(beta_dist.ppf(1-d, k+1, n-k)) if k < n else 1.0
def thresh(s, l, a):
    o = np.argsort(s); ss, ll = s[o], l[o]
    cl = np.cumsum(ll); cn = np.arange(1, len(ss)+1)
    ok = (cl+1)/(cn+1) <= a
    return ss[np.where(ok)[0].max()] if ok.any() else -np.inf
def riskbin(pc, yc, pt, yt, a, nb=8):
    qs = np.quantile(pc, np.linspace(0,1,nb+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    cert=[b for b in range(nb) if ((pc>qs[b])&(pc<=qs[b+1])).sum()>0
          and cp_upper(int(yc[(pc>qs[b])&(pc<=qs[b+1])].sum()), int(((pc>qs[b])&(pc<=qs[b+1])).sum()))<=a]
    tb=np.digitize(pt,qs)-1; acc=np.isin(tb,cert)
    return (float(yt[acc].mean()), float(acc.mean())) if acc.sum() else (0.0,0.0)
def evalmarg(s,l,t):
    m=s<=t; return (float(l[m].mean()), float(m.mean())) if m.sum() else (0.0,0.0)

res={"SE-threshold (marginal)":[], "P(True)-threshold (marginal)":[], "ours risk-binned":[]}
for rep in range(150):
    rr=np.random.default_rng(900+rep); sh=rr.permutation(qids)
    calq=set(sh[:len(qids)//2]); teq=set(sh[len(qids)//2:])
    cal=np.array([g in calq for g in groups]); te=np.array([g in teq for g in groups])
    ck=cal & (~is_noisy | (rr.random(len(cal))<0.30)); tk=te & (is_noisy | (rr.random(len(te))<0.50))
    if ck.sum()<30 or tk.sum()<30: continue
    res["SE-threshold (marginal)"].append(evalmarg(se[tk], y[tk], thresh(se[ck], y[ck], ALPHA)))
    res["P(True)-threshold (marginal)"].append(evalmarg((1-ptrue)[tk], y[tk], thresh((1-ptrue)[ck], y[ck], ALPHA)))
    res["ours risk-binned"].append(riskbin(p_ours[ck], y[ck], p_ours[tk], y[tk], ALPHA))
print("\n=== Selective generation under noise shift (alpha=0.10) ===")
print(f"{'method':30s} {'risk':>6s} {'viol%':>6s} {'answer%':>8s}")
out={"auc":auc}
for k,v in res.items():
    a=np.array(v);
    if len(a)==0: continue
    out[k]=dict(risk=float(a[:,0].mean()), viol=float((a[:,0]>ALPHA).mean()), cov=float(a[:,1].mean()))
    print(f"{k:30s} {a[:,0].mean():6.3f} {(a[:,0]>ALPHA).mean()*100:6.0f} {a[:,1].mean()*100:8.1f}")
json.dump(out, open(f"{DATA}/baselines_report.json","w"), indent=1)
print("\nsaved baselines_report.json")
