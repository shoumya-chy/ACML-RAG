"""Analysis tailored to the Qwen-7B GPU run (feature set: retrieval descriptors,
white-box logprob/entropy, NLI semantic entropy). Risk estimator AUROC by
family, conformal-under-shift (marginal vs risk-binned), and the SE baseline.
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
import sys
TAG=sys.argv[1] if len(sys.argv)>1 else "qwen2_5_7b_instruct"
rows=json.load(open(f"{DATA}/results_{TAG}.json"))
RET=["ret_mean","ret_top1","ret_margin","ret_min","ret_std","has_ctx"]
MOD=["lp_mean","lp_min","ent_mean","ent_max","ent_first","ans_len"]
ALL=RET+MOD
ans=[r for r in rows if r["label"]!="abstain"]
y=np.array([1 if r["label"]=="hallucination" else 0 for r in ans])
groups=np.array([r["qid"] for r in ans]); qids=sorted(set(groups))
se=np.array([r["se"] for r in ans]); ALPHA=0.10
print(f"answered={len(y)} halluc_rate={y.mean():.3f}")

def oof(keys):
    X=np.array([[r[k] for k in keys] for r in ans]); p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups):
        m=GradientBoostingClassifier(random_state=0,n_estimators=150,max_depth=3)
        m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    return p
P={}
for nm,keys in [("retrieval",RET),("model",MOD),("combined",ALL)]:
    P[nm]=oof(keys); print(f"AUROC {nm:10s} {roc_auc_score(y,P[nm]):.3f}")
# +se
Xse=np.array([[r[k] for k in ALL]+[r["se"]] for r in ans]); pse=np.zeros(len(y))
for tr,te in GroupKFold(5).split(Xse,y,groups):
    m=GradientBoostingClassifier(random_state=0,n_estimators=150,max_depth=3); m.fit(Xse[tr],y[tr]); pse[te]=m.predict_proba(Xse[te])[:,1]
print(f"AUROC combined+se {roc_auc_score(y,pse):.3f}")
print(f"AUROC semantic_entropy alone {roc_auc_score(y,se):.3f}")
rng=np.random.default_rng(0); g=[]
for _ in range(2000):
    bq=set(rng.choice(qids,len(qids),replace=True)); m=np.array([q in bq for q in groups])
    if len(set(y[m]))<2: continue
    g.append(roc_auc_score(y[m],P["combined"][m])-roc_auc_score(y[m],se[m]))
print(f"combined - SE: {np.mean(g):+.3f} [95% CI {np.percentile(g,2.5):+.3f}, {np.percentile(g,97.5):+.3f}]")

# conformal under shift
p_all=P["combined"]
noisy={"irr75","irr100","contra_r1","contra_r4","contra_only","mixed"}
is_noisy=np.array([r["cond"] in noisy for r in ans])
def cp_upper(k,n,d=0.10):
    if n==0: return 1.0
    return float(beta_dist.ppf(1-d,k+1,n-k)) if k<n else 1.0
def thr(s,l,a):
    o=np.argsort(s); ss,ll=s[o],l[o]; cl=np.cumsum(ll); cn=np.arange(1,len(ss)+1)
    ok=(cl+1)/(cn+1)<=a; return ss[np.where(ok)[0].max()] if ok.any() else -np.inf
def rb(pc,yc,pt,yt,a,nb=10):
    qs=np.quantile(pc,np.linspace(0,1,nb+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    cert=[b for b in range(nb) if ((pc>qs[b])&(pc<=qs[b+1])).sum()>0 and cp_upper(int(yc[(pc>qs[b])&(pc<=qs[b+1])].sum()),int(((pc>qs[b])&(pc<=qs[b+1])).sum()))<=a]
    tb=np.digitize(pt,qs)-1; acc=np.isin(tb,cert)
    return (float(yt[acc].mean()),float(acc.mean())) if acc.sum() else (0.0,0.0)
def ev(s,l,t):
    m=s<=t; return (float(l[m].mean()),float(m.mean())) if m.sum() else (0.0,0.0)
res={"marginal":[],"riskbin":[]}
for rep in range(200):
    rr=np.random.default_rng(900+rep); sh=rr.permutation(qids)
    cal=np.array([q in set(sh[:len(qids)//2]) for q in groups]); te=np.array([q in set(sh[len(qids)//2:]) for q in groups])
    ck=cal&(~is_noisy|(rr.random(len(cal))<0.30)); tk=te&(is_noisy|(rr.random(len(te))<0.50))
    if ck.sum()<40 or tk.sum()<40: continue
    res["marginal"].append(ev(p_all[tk],y[tk],thr(p_all[ck],y[ck],ALPHA)))
    res["riskbin"].append(rb(p_all[ck],y[ck],p_all[tk],y[tk],ALPHA))
print("\n=== Conformal under noise shift (alpha=0.10) ===")
out={"model":TAG,"answered":len(y),"halluc_rate":float(y.mean()),
     "auc_retrieval":float(roc_auc_score(y,P["retrieval"])),
     "auc_model":float(roc_auc_score(y,P["model"])),
     "auc_combined":float(roc_auc_score(y,P["combined"])),
     "auc_combined_se":float(roc_auc_score(y,pse)),
     "auc_se":float(roc_auc_score(y,se)),
     "combined_minus_se":[float(np.mean(g)),float(np.percentile(g,2.5)),float(np.percentile(g,97.5))]}
for k in res:
    a=np.array(res[k]);
    if len(a)==0: continue
    out[f"crc_{k}"]=dict(risk=float(a[:,0].mean()),viol=float((a[:,0]>ALPHA).mean()),cov=float(a[:,1].mean()))
    print(f"{k:9s} risk={a[:,0].mean():.3f} viol={(a[:,0]>ALPHA).mean():.2f} answer={a[:,1].mean():.3f}")
json.dump(out,open(f"{DATA}/report_{TAG}.json","w"),indent=1)
print(f"saved report_{TAG}.json")
