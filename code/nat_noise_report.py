"""Natural-noise analysis (SQuAD full ~2000-paragraph pool, NO injection).
Pools BM25+dense for power. Tests: does hallucination spike when the retriever
misses the gold paragraph, and does the noise-conditional estimator predict it?
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
RET=["ret_mean","ret_top1","ret_margin","ret_min","ret_std","has_ctx"]
MOD=["lp_mean","lp_min","ent_mean","ent_max","ent_first","ans_len"]
ALL=RET+MOD
rows=[]
for retr in ["bm25","dense"]:
    rs=json.load(open(f"{DATA}/results_natural_{retr}_qwen2_5_7b_instruct.json"))
    for r in rs: r["qid2"]=f"{retr}_{r['qid']}"
    rows+=rs
ans=[r for r in rows if r["label"]!="abstain"]
y=np.array([1 if r["label"]=="hallucination" else 0 for r in ans])
groups=np.array([r["qid2"] for r in ans]); se=np.array([r["se"] for r in ans])
print(f"pooled answered={len(y)} halluc_rate={y.mean():.3f} positives={int(y.sum())}")

# core natural-noise result: hallucination vs whether gold paragraph was retrieved
ret=[r for r in ans if r["cond"]=="retrieved"]
gin=np.array([r.get("gold_in_ctx",0.0) for r in ret]); yr=np.array([1 if r["label"]=="hallucination" else 0 for r in ret])
print(f"\nRETRIEVED subset n={len(ret)}")
print(f"  gold retrieved (gold_in_ctx=1): n={int((gin==1).sum())} halluc={yr[gin==1].mean():.3f}")
print(f"  gold MISSED   (gold_in_ctx=0): n={int((gin==0).sum())} halluc={yr[gin==0].mean():.3f}")
cb=[r for r in ans if r["cond"]=="closedbook"]
if cb: print(f"  closedbook (answered)        : n={len(cb)} halluc={np.mean([1 if r['label']=='hallucination' else 0 for r in cb]):.3f}")

def oof(keys):
    X=np.array([[r[k] for k in keys] for r in ans]); p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups):
        m=GradientBoostingClassifier(random_state=0,n_estimators=150,max_depth=3)
        m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    return p
print("\nEstimator AUROC (question-grouped 5-fold CV):")
P={}
for nm,keys in [("retrieval",RET),("model",MOD),("combined",ALL)]:
    P[nm]=oof(keys); print(f"  {nm:10s} {roc_auc_score(y,P[nm]):.3f}")
print(f"  semantic_entropy alone {roc_auc_score(y,se):.3f}")
qids=sorted(set(groups.tolist())); rng=np.random.default_rng(0); g=[]
for _ in range(2000):
    bq=set(rng.choice(qids,len(qids),replace=True)); m=np.array([q in bq for q in groups])
    if len(set(y[m]))<2: continue
    g.append(roc_auc_score(y[m],P["combined"][m])-roc_auc_score(y[m],se[m]))
print(f"  combined - SE: {np.mean(g):+.3f} [95% CI {np.percentile(g,2.5):+.3f}, {np.percentile(g,97.5):+.3f}]")
out=dict(answered=len(y),halluc_rate=float(y.mean()),
         halluc_gold_retrieved=float(yr[gin==1].mean()) if (gin==1).any() else None,
         halluc_gold_missed=float(yr[gin==0].mean()) if (gin==0).any() else None,
         n_gold_retrieved=int((gin==1).sum()),n_gold_missed=int((gin==0).sum()),
         auc_retrieval=float(roc_auc_score(y,P["retrieval"])),auc_model=float(roc_auc_score(y,P["model"])),
         auc_combined=float(roc_auc_score(y,P["combined"])),auc_se=float(roc_auc_score(y,se)),
         combined_minus_se=[float(np.mean(g)),float(np.percentile(g,2.5)),float(np.percentile(g,97.5))])
json.dump(out,open(f"{DATA}/natnoise_report.json","w"),indent=1)
print("\nsaved natnoise_report.json")
