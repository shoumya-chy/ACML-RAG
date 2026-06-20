"""Advanced analyses for the journal version:
 (A) Multi-model conformal comparison (one model per call via TAG arg).
 (B) Isotonic recalibration of the risk score -> answer-rate gain for risk-binned CRC.
 (C) Estimator-architecture ablation (logreg / RF / GBM / MLP).
Writes data/advanced_<TAG>.json and prints tables.

Usage: python3 analyze_advanced.py <TAG>     e.g. flan_t5_base | flan_t5_small | qwen2_5_0_5b
"""
import json, os, sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TAG = sys.argv[1] if len(sys.argv) > 1 else "flan_t5_base"
NREPS = int(os.environ.get("NREPS", "100"))
ALPHA = 0.10
rng = np.random.default_rng(11)

RET = ["ret_mean", "ret_top1", "ret_margin", "ret_min", "ret_std", "conflict", "n_claims", "has_ctx"]
MOD = ["lp_mean", "lp_min", "ent_mean", "ent_max", "ent_first", "ans_len"]
DELTA = ["d_lp", "d_ent", "ans_changed"]
ALL = RET + MOD + DELTA

rows = json.load(open(f"{DATA}/results_{TAG}.json"))
answered = [r for r in rows if r["label"] != "abstain"]
groups = np.array([r["qid"] for r in answered])
y = np.array([1 if r["label"] == "hallucination" else 0 for r in answered])
X = np.array([[r[k] for k in ALL] for r in answered])
qids = sorted(set(groups))
report = {"model": TAG, "n_answered": len(y), "base_rate": float(y.mean())}

# whether this model has decoder-only single-class issues: use available features
HAS_MOD = np.std([r["lp_mean"] for r in answered]) > 1e-9  # ollama/causal may lack logprobs
FEATS = ALL if HAS_MOD else (RET + ["ans_len", "ans_changed"])

def oof(estimator_fn, feats):
    Xf = np.array([[r[k] for k in feats] for r in answered])
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xf, y, groups):
        m = estimator_fn(); m.fit(Xf[tr], y[tr]); p[te] = m.predict_proba(Xf[te])[:, 1]
    return p

def gbm(): return GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3)
def rf(): return RandomForestClassifier(random_state=0, n_estimators=300, n_jobs=2)
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
def mlp(): return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16),
                                max_iter=600, random_state=0))

# ---- (C) estimator ablation ----
print(f"== {TAG}  (answered n={len(y)}, base rate {y.mean():.3f}, model-feats={HAS_MOD}) ==")
print("\n=== Estimator-architecture ablation (combined features, grouped CV) ===")
report["estimator_ablation"] = {}
p_main = None
for name, fn in [("logreg", lr), ("random_forest", rf), ("gbm", gbm), ("mlp", mlp)]:
    p = oof(fn, FEATS)
    a = roc_auc_score(y, p)
    report["estimator_ablation"][name] = float(a)
    print(f"{name:14s} AUROC={a:.3f}")
    if name == "gbm":
        p_main = p

# ---- (B) isotonic recalibration ----
def crc_threshold(scores, labels, alpha):
    order = np.argsort(scores); s, l = scores[order], labels[order]
    cl = np.cumsum(l); cn = np.arange(1, len(s) + 1)
    ok = (cl + 1) / (cn + 1) <= alpha
    return s[np.where(ok)[0].max()] if ok.any() else -np.inf

def cp_upper(k, n, delta=0.10):
    if n == 0: return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k)) if k < n else 1.0

def riskbin(pc, yc, pt, yt, alpha, nbins=10):
    qs = np.quantile(pc, np.linspace(0, 1, nbins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cert = []
    for b in range(nbins):
        m = (pc > qs[b]) & (pc <= qs[b + 1])
        if m.sum() > 0 and cp_upper(int(yc[m].sum()), int(m.sum())) <= alpha:
            cert.append(b)
    tb = np.digitize(pt, qs) - 1
    acc = np.isin(tb, cert)
    if acc.sum() == 0: return 0.0, 0.0
    return float(yt[acc].mean()), float(acc.mean())

noisy = {"irr75", "irr100", "contra_r1", "contra_r4", "contra_only", "mixed"}
is_noisy = np.array([r["cond"] in noisy for r in answered])

def splits(rep):
    rr = np.random.default_rng(7000 + rep)
    qsh = rr.permutation(qids)
    calq = set(qsh[:len(qids) // 2]); teq = set(qsh[len(qids) // 2:])
    cal = np.array([g in calq for g in groups]); te = np.array([g in teq for g in groups])
    ck = cal & (~is_noisy | (rr.random(len(cal)) < 0.30))
    tk = te & (is_noisy | (rr.random(len(te)) < 0.50))
    return ck, tk

print("\n=== Isotonic recalibration effect on risk-binned CRC ===")
raw_r, raw_c, iso_r, iso_c = [], [], [], []
for rep in range(NREPS):
    ck, tk = splits(rep)
    if ck.sum() < 40 or tk.sum() < 40: continue
    r0, c0 = riskbin(p_main[ck], y[ck], p_main[tk], y[tk], ALPHA)
    raw_r.append(r0); raw_c.append(c0)
    # fit isotonic on calibration only, apply to both
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_main[ck], y[ck])
    pcc, ptt = iso.predict(p_main[ck]), iso.predict(p_main[tk])
    r1, c1 = riskbin(pcc, y[ck], ptt, y[tk], ALPHA)
    iso_r.append(r1); iso_c.append(c1)
report["recal_raw"] = dict(risk=float(np.mean(raw_r)), viol=float(np.mean(np.array(raw_r) > ALPHA)),
                           cov=float(np.mean(raw_c)))
report["recal_iso"] = dict(risk=float(np.mean(iso_r)), viol=float(np.mean(np.array(iso_r) > ALPHA)),
                           cov=float(np.mean(iso_c)))
print(f"raw      risk={np.mean(raw_r):.3f} viol={np.mean(np.array(raw_r)>ALPHA):.2f} answer={np.mean(raw_c):.3f}")
print(f"isotonic risk={np.mean(iso_r):.3f} viol={np.mean(np.array(iso_r)>ALPHA):.2f} answer={np.mean(iso_c):.3f}")

# ---- (A) full method comparison for this model ----
def marginal(pc, yc, pt, yt, alpha):
    t = crc_threshold(pc, yc, alpha); m = pt <= t
    return (float(yt[m].mean()), float(m.mean())) if m.sum() else (0.0, 0.0)

print("\n=== Conformal method comparison (this model) ===")
mm = {"marginal": [], "riskbin": []}
for rep in range(NREPS):
    ck, tk = splits(rep)
    if ck.sum() < 40 or tk.sum() < 40: continue
    mm["marginal"].append(marginal(p_main[ck], y[ck], p_main[tk], y[tk], ALPHA))
    mm["riskbin"].append(riskbin(p_main[ck], y[ck], p_main[tk], y[tk], ALPHA))
for k in mm:
    a = np.array(mm[k])
    report[f"crc_{k}"] = dict(risk=float(a[:, 0].mean()), viol=float((a[:, 0] > ALPHA).mean()),
                              cov=float(a[:, 1].mean()))
    print(f"{k:9s} risk={a[:,0].mean():.3f} viol={(a[:,0]>ALPHA).mean():.2f} answer={a[:,1].mean():.3f}")

json.dump(report, open(f"{DATA}/advanced_{TAG}.json", "w"), indent=1)
print("\nsaved advanced json")
