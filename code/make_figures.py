"""Generate publication figures for the new results. Saves PNGs to ../paper/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
PAP=os.path.abspath(os.path.join(HERE,"..","paper"))
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
C={"se":"#c0504d","ret":"#4f81bd","comb":"#9bbb59","alt":"#8064a2"}

# ---------- FIG 1: missing-evidence vs misinformation decomposition ----------
def auc_by_group(tag):
    z=np.load(f"{DATA}/cache_{tag}.npz",allow_pickle=True)
    p_ret,p_all,se,y,cond=z["p_ret"],z["p_all"],z["se"],z["y"],z["cond"]
    G={"Missing-evidence\n(closed-book, irrelevance)":{"closedbook","irr50","irr100"},
       "Misinformation\n(contradiction, poison)":{"contra_r1","contra_only","poison","mixed"}}
    out={}
    for name,cs in G.items():
        m=np.array([c in cs for c in cond])
        if len(set(y[m]))<2: continue
        out[name]=(roc_auc_score(y[m],se[m]),roc_auc_score(y[m],p_ret[m]),roc_auc_score(y[m],p_all[m]))
    return out
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,(tag,title) in zip(axes,[("squad_bm25_qwen2_5_7b_instruct","Qwen2.5-7B"),("squad_bm25_mistral_7b_instruct_v0_3","Mistral-7B-v0.3")]):
    d=auc_by_group(tag); groups=list(d); x=np.arange(len(groups)); w=0.26
    se=[d[g][0] for g in groups]; rt=[d[g][1] for g in groups]; cb=[d[g][2] for g in groups]
    ax.bar(x-w,se,w,label="Semantic entropy",color=C["se"])
    ax.bar(x,rt,w,label="Retrieval descriptors",color=C["ret"])
    ax.bar(x+w,cb,w,label="Combined",color=C["comb"])
    ax.axhline(0.5,ls="--",lw=0.8,color="gray")
    ax.set_xticks(x); ax.set_xticklabels(groups,fontsize=9.5); ax.set_title(title)
    ax.set_ylim(0.35,0.95)
    for xi,v in zip(x-w,se): ax.text(xi,v+0.01,f"{v:.2f}",ha="center",fontsize=8)
    for xi,v in zip(x,rt): ax.text(xi,v+0.01,f"{v:.2f}",ha="center",fontsize=8)
    for xi,v in zip(x+w,cb): ax.text(xi,v+0.01,f"{v:.2f}",ha="center",fontsize=8)
axes[0].set_ylabel("Hallucination-prediction AUROC")
axes[0].legend(loc="upper left",fontsize=9,framealpha=0.9)
fig.tight_layout(); fig.savefig(f"{PAP}/fig_decomposition.png",bbox_inches="tight"); plt.close(fig)
print("wrote fig_decomposition.png")

# ---------- FIG 2: six-family panel (combined vs SE AUROC) ----------
order=["qwen2_5_1_5b_instruct","qwen2_5_3b_instruct","phi_3_5_mini_instruct","mistral_7b_instruct_v0_3","qwen2_5_7b_instruct","gemma_2_9b_it"]
disp=["Qwen2.5\n1.5B","Qwen2.5\n3B","Phi-3.5\n3.8B","Mistral\n7B","Qwen2.5\n7B","Gemma-2\n9B"]
comb=[];se=[]
for t in order:
    d=json.load(open(f"{DATA}/report_{t}.json")); comb.append(d["auc_combined"]); se.append(d["auc_se"])
x=np.arange(len(order)); w=0.38
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(x-w/2,comb,w,label="Combined estimator",color=C["comb"])
ax.bar(x+w/2,se,w,label="Semantic entropy",color=C["se"])
ax.axhline(0.5,ls="--",lw=0.8,color="gray")
for xi,v in zip(x-w/2,comb): ax.text(xi,v+0.01,f"{v:.2f}",ha="center",fontsize=8)
for xi,v in zip(x+w/2,se): ax.text(xi,v+0.01,f"{v:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(disp,fontsize=9); ax.set_ylim(0.4,1.0)
ax.set_ylabel("Hallucination-prediction AUROC"); ax.set_title("Generality across six model families (controlled grid)")
ax.legend(loc="upper center",ncol=2,fontsize=9)
fig.tight_layout(); fig.savefig(f"{PAP}/fig_sixfamily.png",bbox_inches="tight"); plt.close(fig)
print("wrote fig_sixfamily.png")

# ---------- FIG 3: naturalistic per-condition hallucination (Qwen + Mistral, BM25) ----------
conds=["clean","contra_r1","mixed","irr100","contra_only","poison","irr50","closedbook"]
def pc(tag):
    d=json.load(open(f"{DATA}/natreport_{tag}.json"))["per_condition_halluc"]; return [d.get(c,0) for c in conds]
q=pc("squad_bm25_qwen2_5_7b_instruct"); m=pc("squad_bm25_mistral_7b_instruct_v0_3")
x=np.arange(len(conds)); w=0.4
fig,ax=plt.subplots(figsize=(9.5,4))
ax.bar(x-w/2,q,w,label="Qwen2.5-7B",color=C["ret"]); ax.bar(x+w/2,m,w,label="Mistral-7B",color=C["alt"])
ax.set_xticks(x); ax.set_xticklabels(conds,rotation=30,ha="right",fontsize=9)
ax.set_ylabel("Hallucination rate (answered)"); ax.set_ylim(0,1.0)
ax.set_title("Naturalistic SQuAD: hallucination by noise condition (BM25)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{PAP}/fig_naturalistic_percond.png",bbox_inches="tight"); plt.close(fig)
print("wrote fig_naturalistic_percond.png")

# ---------- FIG 4: natural-noise gold-in-context ----------
d=json.load(open(f"{DATA}/natnoise_report.json"))
labels=["Gold paragraph\nretrieved","Gold paragraph\nMISSED","Closed-book"]
vals=[d["halluc_gold_retrieved"],d["halluc_gold_missed"],0.533]
fig,ax=plt.subplots(figsize=(6,4))
bars=ax.bar(labels,vals,color=[C["comb"],C["se"],"#999999"],width=0.6)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.01,f"{v:.2f}",ha="center",fontsize=10)
ax.set_ylabel("Hallucination rate (answered)"); ax.set_ylim(0,0.75)
ax.set_title("Natural retrieval noise (no injection): retrieval miss drives hallucination")
fig.tight_layout(); fig.savefig(f"{PAP}/fig_natnoise.png",bbox_inches="tight"); plt.close(fig)
print("wrote fig_natnoise.png")
print("ALL FIGURES DONE ->",PAP)
