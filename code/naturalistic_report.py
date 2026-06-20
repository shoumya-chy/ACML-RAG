"""Naturalistic (SQuAD real-retrieval) analysis from cached OOF predictions.
Noise-shift set adapted to the naturalistic conditions. Reports per-condition
hallucination, estimator AUROC, SE gap, and marginal-vs-Mondrian conformal.
Usage: python3 naturalistic_report.py <tag>
"""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
TAG=sys.argv[1]; ALPHA=0.10
z=np.load(f"{DATA}/cache_{TAG}.npz",allow_pickle=True)
p_ret,p_mod,p_all,pse=z["p_ret"],z["p_mod"],z["p_all"],z["pse"]
y,groups,se,cond=z["y"],z["groups"],z["se"],z["cond"]
qids=sorted(set(groups.tolist()))
print(f"{TAG} answered={len(y)} halluc_rate={y.mean():.3f}")
# per-condition hallucination among answered
rows=json.load(open(f"{DATA}/results_{TAG}.json"))
ans=[r for r in rows if r["label"]!="abstain"]
pc={}
for r in ans: pc.setdefault(r["cond"],[]).append(1 if r["label"]=="hallucination" else 0)
ab={}
for r in rows: ab.setdefault(r["cond"],[]).append(1 if r["label"]=="abstain" else 0)
print("per-condition  halluc(ans) / abstain(all):")
for c in ["closedbook","clean","irr50","irr100","contra_r1","contra_only","poison","mixed"]:
    if c in pc: print(f"  {c:12s} h={np.mean(pc[c]):.3f} (n={len(pc[c])})  abst={np.mean(ab[c]):.3f}")
print(f"AUROC retrieval {roc_auc_score(y,p_ret):.3f}  model {roc_auc_score(y,p_mod):.3f}  combined {roc_auc_score(y,p_all):.3f}")
print(f"AUROC combined+se {roc_auc_score(y,pse):.3f}  se_alone {roc_auc_score(y,se):.3f}")

rng=np.random.default_rng(0); g=[]
for _ in range(2000):
    bq=set(rng.choice(qids,len(qids),replace=True)); m=np.array([q in bq for q in groups])
    if len(set(y[m]))<2: continue
    g.append(roc_auc_score(y[m],p_all[m])-roc_auc_score(y[m],se[m]))
print(f"combined - SE: {np.mean(g):+.3f} [95% CI {np.percentile(g,2.5):+.3f}, {np.percentile(g,97.5):+.3f}]")

# conformal under naturalistic noise shift
noisy={"irr50","irr100","contra_r1","contra_only","poison","mixed"}
is_noisy=np.array([c in noisy for c in cond])
def cp_upper(k,n,d=0.10):
    if n==0: return 1.0
    return float(beta_dist.ppf(1-d,k+1,n-k)) if k<n else 1.0
def thr(s,l,a):
    o=np.argsort(s); ss,ll=s[o],l[o]; cl=np.cumsum(ll); cn=np.arange(1,len(ss)+1)
    ok=(cl+1)/(cn+1)<=a; return ss[np.where(ok)[0].max()] if ok.any() else -np.inf
def rb(pcal,yc,pt,yt,a,nb=10):
    qs=np.quantile(pcal,np.linspace(0,1,nb+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    cert=[b for b in range(nb) if ((pcal>qs[b])&(pcal<=qs[b+1])).sum()>0 and cp_upper(int(yc[(pcal>qs[b])&(pcal<=qs[b+1])].sum()),int(((pcal>qs[b])&(pcal<=qs[b+1])).sum()))<=a]
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
out={"model":TAG,"answered":int(len(y)),"halluc_rate":float(y.mean()),
     "auc_retrieval":float(roc_auc_score(y,p_ret)),"auc_model":float(roc_auc_score(y,p_mod)),
     "auc_combined":float(roc_auc_score(y,p_all)),"auc_combined_se":float(roc_auc_score(y,pse)),
     "auc_se":float(roc_auc_score(y,se)),
     "combined_minus_se":[float(np.mean(g)),float(np.percentile(g,2.5)),float(np.percentile(g,97.5))],
     "per_condition_halluc":{c:float(np.mean(v)) for c,v in pc.items()}}
print("\n=== Conformal under naturalistic noise shift (alpha=0.10) ===")
for k in res:
    a=np.array(res[k])
    if len(a)==0: continue
    out[f"crc_{k}"]=dict(risk=float(a[:,0].mean()),viol=float((a[:,0]>ALPHA).mean()),cov=float(a[:,1].mean()))
    print(f"{k:9s} risk={a[:,0].mean():.3f} viol={(a[:,0]>ALPHA).mean():.2f} answer={a[:,1].mean():.3f}")
json.dump(out,open(f"{DATA}/natreport_{TAG}.json","w"),indent=1)
print(f"saved natreport_{TAG}.json")
