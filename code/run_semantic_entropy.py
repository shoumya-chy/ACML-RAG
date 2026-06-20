"""Real UQ baselines: discrete semantic entropy (Kuhn/Farquhar style) and
P(True) (Kadavath style), computed on the SAME generator so the comparison
is apples-to-apples. For short closed-form answers, exact normalised-string
equivalence is the semantic-equivalence relation, so discrete semantic
entropy is exact rather than approximated by an NLI model.

Writes data/results_se_<model>.json with per-(q,cond) records carrying:
  greedy answer + label, se_discrete (5 samples), p_true, plus retrieval
  descriptors so the records merge with the main feature set.

Sandbox-local; reads DATA from env, writes checkpoints to /tmp/pilot.
Usage: HF_HUB_OFFLINE=1 python3 run_semantic_entropy.py <budget_s> [conds]
"""
import json, os, re, sys, time, random, math
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
MODEL = os.environ.get("PILOT_MODEL", "google/flan-t5-small")
TAG = "se_" + re.sub(r"[^a-zA-Z0-9]+", "_", MODEL.split("/")[-1])
SUBSET = int(os.environ.get("SUBSET", "4"))
NSAMP = int(os.environ.get("NSAMP", "5"))
CKPT = "/tmp/pilot"; os.makedirs(CKPT, exist_ok=True)

docs = json.load(open(f"{DATA}/docs.json"))
questions_all = json.load(open(f"{DATA}/questions.json"))
questions = [q for i, q in enumerate(questions_all)
             if q["regime"] == "real" or (i % SUBSET == 0)]
bm25 = BM25Okapi([re.findall(r"\w+", d["text"].lower()) for d in docs])
by_kind = {}
for i, d in enumerate(docs):
    by_kind.setdefault(d["kind"], []).append(i)
gold_idx, contra_idx, distract_idx = {}, {}, {}
for i, d in enumerate(docs):
    k = (d.get("entity"), d.get("attr"))
    if d["kind"] == "gold": gold_idx[k] = i
    elif d["kind"] == "contra": contra_idx[k] = i
    elif d["kind"] == "distractor": distract_idx[(d.get("target"), d.get("attr"))] = i

tk = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL); model.eval(); torch.set_num_threads(2)
print("loaded", MODEL, "questions", len(questions), flush=True)

def retrieve_clean(q, k=4):
    sc = bm25.get_scores(re.findall(r"\w+", q.lower()))
    order = np.argsort(-sc)
    return [int(i) for i in order if docs[i]["kind"] not in ("contra", "distractor")][:k], sc

def build_condition(q, cond, rng):
    if cond == "closedbook": return [], None
    base, sc = retrieve_clean(q["question"])
    gi = gold_idx[(q["entity"], q["attr"])]
    if gi not in base: base = [gi] + base[:3]
    ci = contra_idx.get((q["entity"], q["attr"]))
    if cond == "clean": ctx = list(base); rng.shuffle(ctx)
    elif cond == "irr50": ctx = base[:2] + rng.sample(by_kind["irrelevant"], 2); rng.shuffle(ctx)
    elif cond == "irr100": ctx = rng.sample(by_kind["irrelevant"], 4)
    elif cond == "contra_r1": ctx = [ci] + [d for d in base if d != ci][:3]
    elif cond == "contra_only": ctx = [ci] + rng.sample(by_kind["irrelevant"], 3); rng.shuffle(ctx)
    else: ctx = list(base)
    return ctx, sc

def prompt_for(q, ctx_ids):
    if ctx_ids:
        ctx = "\n".join(f"- {docs[i]['text']}" for i in ctx_ids)
        return (f"Answer using the context. If the context does not contain the "
                f"answer, answer unknown.\nContext:\n{ctx}\nQ: {q} A:")
    return f"Q: {q} A:"

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
ABST = re.compile(r"(don.?t know|unknown|cannot|not (mentioned|stated|provided|given)|no answer)")
def label(ans, gold):
    a, g = norm(ans), norm(gold)
    if not a or ABST.search(a): return "abstain"
    if g in a or (a in g and len(a) > 2): return "correct"
    if re.fullmatch(r"\d{4}", g) and g in re.findall(r"\d{4}", a): return "correct"
    return "hallucination"

@torch.no_grad()
def greedy(prompt):
    enc = tk(prompt, return_tensors="pt", truncation=True, max_length=512)
    o = model.generate(**enc, max_new_tokens=12, do_sample=False)
    return tk.decode(o[0], skip_special_tokens=True).strip()

@torch.no_grad()
def sample(prompt, n):
    enc = tk(prompt, return_tensors="pt", truncation=True, max_length=512)
    outs = model.generate(**enc, max_new_tokens=12, do_sample=True, temperature=0.7,
                          top_p=0.9, num_return_sequences=n)
    return [tk.decode(o, skip_special_tokens=True).strip() for o in outs]

def discrete_semantic_entropy(samples):
    # exact-match clustering = semantic equivalence for short closed answers
    from collections import Counter
    c = Counter(norm(s) for s in samples)
    tot = sum(c.values())
    return float(-sum((v / tot) * math.log(v / tot) for v in c.values())), len(c)

@torch.no_grad()
def p_true(q, ctx_ids, ans):
    # Kadavath-style: ask model whether the proposed answer is true; read P(True)
    if ctx_ids:
        ctx = "\n".join(f"- {docs[i]['text']}" for i in ctx_ids)
        pr = (f"Context:\n{ctx}\nQuestion: {q}\nProposed answer: {ans}\n"
              f"Is the proposed answer correct? Answer True or False. Answer:")
    else:
        pr = (f"Question: {q}\nProposed answer: {ans}\n"
              f"Is the proposed answer correct? Answer True or False. Answer:")
    enc = tk(pr, return_tensors="pt", truncation=True, max_length=512)
    dec = tk("<pad>", return_tensors="pt").input_ids[:, :1]
    out = model(**enc, decoder_input_ids=dec)
    logits = out.logits[0, -1]
    # ids for 'True'/'False' first tokens
    tid = tk("True", add_special_tokens=False).input_ids[0]
    fid = tk("False", add_special_tokens=False).input_ids[0]
    lp = torch.log_softmax(logits, -1)
    pt = float(torch.exp(lp[tid])); pf = float(torch.exp(lp[fid]))
    z = pt + pf + 1e-9
    return pt / z

CONDS = os.environ.get("CONDS", "closedbook,clean,irr50,irr100,contra_r1,contra_only").split(",")

def atomic(obj, path):
    t = path + ".tmp"; json.dump(obj, open(t, "w")); os.replace(t, path)

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    part = f"{CKPT}/results_{TAG}_partial.json"
    rows = json.load(open(part)) if os.path.exists(part) else []
    done = {(r["qid"], r["cond"]) for r in rows}
    t0 = time.time()
    for qi, q in enumerate(questions):
        rng = random.Random(4000 + qi)
        for cond in CONDS:
            if (q["qid"], cond) in done: continue
            if time.time() - t0 > budget:
                atomic(rows, part); print(f"CHECKPOINT rows={len(rows)}", flush=True); return
            ctx, sc = build_condition(q, cond, rng)
            pr = prompt_for(q["question"], ctx)
            g = greedy(pr); lab = label(g, q["gold"])
            sm = sample(pr, NSAMP); se, nclu = discrete_semantic_entropy(sm)
            pt = p_true(q["question"], ctx, g)
            if ctx:
                cs = sorted([float(sc[i]) for i in ctx], reverse=True)
                feat = dict(ret_mean=float(np.mean(cs)), ret_top1=cs[0],
                            ret_margin=cs[0]-cs[1], ret_min=cs[-1], has_ctx=1.0)
            else:
                feat = dict(ret_mean=0, ret_top1=0, ret_margin=0, ret_min=0, has_ctx=0.0)
            rows.append(dict(qid=q["qid"], regime=q["regime"], cond=cond, gold=q["gold"],
                             answer=g, label=lab, se=se, se_nclusters=nclu, p_true=pt, **feat))
            done.add((q["qid"], cond))
        if qi % 5 == 0:
            print(f"q {qi}/{len(questions)} rows={len(rows)} t={time.time()-t0:.0f}s", flush=True)
            atomic(rows, part)
    atomic(rows, part); atomic(rows, f"{DATA}/results_{TAG}.json")
    print(f"DONE rows={len(rows)} time={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
