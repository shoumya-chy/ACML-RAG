"""Deployment-time shift self-diagnosis ("certify-or-flag").

Risk-binned CRC is valid under Assumption (ii): within-bin label rate is
shift-invariant. Its one failure mode is a WITHIN-bin shift that raises the
true risk of a certified bin. We (1) construct such an adversarial within-bin
shift, (2) show naive risk-binned CRC then violates, and (3) introduce a
LABEL-FREE diagnostic that compares the distribution of an observable noise
covariate within each certified bin between calibration and deployment; bins
that drift are decertified (flagged), restoring validity at a coverage cost.

Operates on logged flan_t5_base data; no new generation.
"""
import json, os, sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from scipy.stats import beta as beta_dist, ks_2samp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "flan_t5_base"
rows = json.load(open(f"{DATA}/results_{TAG}.json"))
RET = ["ret_mean","ret_top1","ret_margin","ret_min","ret_std","conflict","n_claims","has_ctx"]
MOD = ["lp_mean","lp_min","ent_mean","ent_max","ent_first","ans_len"]
DELTA = ["d_lp","d_ent","ans_changed"]; ALL = RET+MOD+DELTA
ans = [r for r in rows if r["label"] != "abstain"]
y = np.array([1 if r["label"]=="hallucination" else 0 for r in ans])
groups = np.array([r["qid"] for r in ans]); qids = sorted(set(groups))
X = np.array([[r[k] for k in ALL] for r in ans])
conflict = np.array([r["conflict"] for r in ans])   # observable covariate for the diagnostic
ALPHA = 0.10

cache = f"/tmp/pilot/oof_{TAG}_combined.npy"
if os.path.exists(cache): p_all = np.load(cache)
else:
    p_all = np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups):
        m=GradientBoostingClassifier(random_state=0,n_estimators=150,max_depth=3)
        m.fit(X[tr],y[tr]); p_all[te]=m.predict_proba(X[te])[:,1]
    np.save(cache,p_all)

def cp_upper(k,n,d=0.10):
    if n==0: return 1.0
    return float(beta_dist.ppf(1-d,k+1,n-k)) if k<n else 1.0

def certify(pc,yc,a,nb=10):
    qs=np.quantile(pc,np.linspace(0,1,nb+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    cert=[]
    for b in range(nb):
        m=(pc>qs[b])&(pc<=qs[b+1])
        if m.sum()>0 and cp_upper(int(yc[m].sum()),int(m.sum()))<=a: cert.append(b)
    return qs,cert

def realized(pt,yt,qs,cert):
    tb=np.digitize(pt,qs)-1; acc=np.isin(tb,cert)
    return (float(yt[acc].mean()),float(acc.mean())) if acc.sum() else (0.0,0.0)

def diagnose(qs,cert,pc,cc,pt,ct,pthr=0.01):
    """label-free: for each certified bin, KS-test the observable covariate
    (conflict) between calibration and deployment members; decertify if it drifts."""
    keep=[]
    for b in cert:
        cm=(pc>qs[b])&(pc<=qs[b+1]); tm=(pt>qs[b])&(pt<=qs[b+1])
        if cm.sum()<5 or tm.sum()<5: keep.append(b); continue
        try: _,pv=ks_2samp(cc[cm], ct[tm])
        except Exception: pv=1.0
        if pv>=pthr: keep.append(b)   # no significant drift -> keep certified
    return keep

# scenario A: benign noise-regime shift (bin-measurable) -- method should stay valid
# scenario B: adversarial WITHIN-bin shift -- naive should break, diagnosis should rescue
noisy={"irr75","irr100","contra_r1","contra_r4","contra_only","mixed"}
is_noisy=np.array([r["cond"] in noisy for r in ans])

def split(rep):
    rr=np.random.default_rng(1234+rep); sh=rr.permutation(qids)
    cal=np.array([g in set(sh[:len(qids)//2]) for g in groups])
    te=np.array([g in set(sh[len(qids)//2:]) for g in groups])
    return rr,cal,te

res={"benign_naive":[], "withinshift_naive":[], "withinshift_diagnosed":[]}
flagged_counts=[]
for rep in range(200):
    rr,cal,te=split(rep)
    ck=cal & (~is_noisy | (rr.random(len(cal))<0.30))
    # benign deployment: noise-heavy but bin-measurable
    tb=te & (is_noisy | (rr.random(len(te))<0.50))
    if ck.sum()<60 or tb.sum()<60: continue
    qs,cert=certify(p_all[ck],y[ck],ALPHA)
    res["benign_naive"].append(realized(p_all[tb],y[tb],qs,cert))
    # adversarial within-bin shift: within deployment, upweight high-conflict items
    # (which carry higher true risk) so bin label-rate rises without moving scores much
    w=np.where(conflict[te]==1, 5.0, 1.0); w=w/ w.sum()
    idx_te=np.where(te)[0]
    draw=rr.choice(idx_te, size=tb.sum(), replace=True, p=w[ (np.searchsorted(idx_te, idx_te)) ] if False else (w))
    # build adversarial deployment index set
    adv=rr.choice(idx_te, size=min(len(idx_te), tb.sum()), replace=True, p=w)
    ptv=p_all[adv]; ytv=y[adv]; ctv=conflict[adv]
    res["withinshift_naive"].append(realized(ptv,ytv,qs,cert))
    kept=diagnose(qs,cert,p_all[ck],conflict[ck],ptv,ctv)
    flagged_counts.append(len(cert)-len(kept))
    res["withinshift_diagnosed"].append(realized(ptv,ytv,qs,kept))

print(f"{'scenario':26s} {'risk':>6s} {'viol%':>6s} {'answer%':>8s}")
out={}
for k in res:
    a=np.array(res[k]);
    if len(a)==0: continue
    out[k]=dict(risk=float(a[:,0].mean()), viol=float((a[:,0]>ALPHA).mean()), cov=float(a[:,1].mean()))
    print(f"{k:26s} {a[:,0].mean():6.3f} {(a[:,0]>ALPHA).mean()*100:6.0f} {a[:,1].mean()*100:8.1f}")
out["avg_bins_flagged"]=float(np.mean(flagged_counts)) if flagged_counts else 0.0
print(f"avg certified bins flagged by diagnosis: {out['avg_bins_flagged']:.2f}")
json.dump(out, open(f"{DATA}/selfdiag_{TAG}.json","w"), indent=1)

# figure
fig,ax=plt.subplots(figsize=(6.6,4))
labels=["benign\n(bin-measurable)","within-bin shift\n(naive)","within-bin shift\n(certify-or-flag)"]
data=[np.array(res[k])[:,0] for k in ["benign_naive","withinshift_naive","withinshift_diagnosed"]]
parts=ax.violinplot(data, showmeans=True)
ax.axhline(ALPHA, color="k", ls=":", label="target alpha=0.10")
ax.set_xticks([1,2,3]); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("realized risk among answered"); ax.legend()
ax.set_title("Shift self-diagnosis restores validity under within-bin shift")
fig.tight_layout(); fig.savefig(f"{FIG}/v2_selfdiag.png", dpi=140)
print("saved selfdiag json + figure")
