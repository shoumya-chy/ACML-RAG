"""Stage 1: compute out-of-fold risk-estimator predictions and cache them.
Identical model/feature settings to analyze_qwen7b.py so downstream numbers match.
Usage: python3 stage1_cache.py <tag>
"""
import json, os, sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
TAG=sys.argv[1]
rows=json.load(open(f"{DATA}/results_{TAG}.json"))
RET=["ret_mean","ret_top1","ret_margin","ret_min","ret_std","has_ctx"]
MOD=["lp_mean","lp_min","ent_mean","ent_max","ent_first","ans_len"]
ALL=RET+MOD
ans=[r for r in rows if r["label"]!="abstain"]
y=np.array([1 if r["label"]=="hallucination" else 0 for r in ans])
groups=np.array([r["qid"] for r in ans])
se=np.array([r["se"] for r in ans])
cond=np.array([r["cond"] for r in ans])

def oof(keys):
    X=np.array([[r[k] for k in keys] for r in ans]); p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups):
        m=GradientBoostingClassifier(random_state=0,n_estimators=150,max_depth=3)
        m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    return p
p_ret=oof(RET); p_mod=oof(MOD); p_all=oof(ALL)
Xse=np.array([[r[k] for k in ALL]+[r["se"]] for r in ans]); pse=np.zeros(len(y))
for tr,te in GroupKFold(5).split(Xse,y,groups):
    m=GradientBoostingClassifier(random_state=0,n_estimators=150,max_depth=3)
    m.fit(Xse[tr],y[tr]); pse[te]=m.predict_proba(Xse[te])[:,1]
np.savez(f"{DATA}/cache_{TAG}.npz",p_ret=p_ret,p_mod=p_mod,p_all=p_all,pse=pse,
         y=y,groups=groups,se=se,cond=cond)
print("cached",TAG,"n=",len(y),"halluc_rate=",round(float(y.mean()),3))
