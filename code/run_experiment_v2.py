"""v2 experiment harness: 11 noise conditions, closedbook-contrast (confidence-gain)
features, resumable checkpoints, model-agnostic (PILOT_MODEL env var).

Conditions:
  clean            top-4 clean BM25 (gold guaranteed reachable)
  irr25/50/75/100  graded irrelevance (1/2/3/4 of 4 docs replaced; irr100 removes gold)
  contra_r1        gold present + contradiction doc at rank 1
  contra_r4        gold present + contradiction doc at last rank (position effect)
  contra_only      contradiction doc WITHOUT gold (pure misinformation)
  distractor       similar-name distractor doc at rank 1 + gold
  mixed            gold + 1 contra + 1 irrelevant + 1 neutral (compound noise)
  closedbook       no context

Usage: python3 run_experiment_v2.py <budget_seconds> [conds_csv]
Env:   PILOT_MODEL=/tmp/pilot/flan-t5-base (default) | google/flan-t5-small
"""
import json, os, re, sys, time, random
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PILOT_DATA", os.path.join(HERE, "data"))
MODEL = os.environ.get("PILOT_MODEL", "/tmp/pilot/flan-t5-base")
TAG = re.sub(r"[^a-zA-Z0-9]+", "_", MODEL.split("/")[-1]) + os.environ.get("TAG_SUFFIX", "")

docs = json.load(open(f"{DATA}/docs.json"))
questions = json.load(open(f"{DATA}/questions.json"))
texts = [d["text"] for d in docs]
bm25 = BM25Okapi([re.findall(r"\w+", t.lower()) for t in texts])
by_kind = {}
for i, d in enumerate(docs):
    by_kind.setdefault(d["kind"], []).append(i)
gold_idx, contra_idx, distract_idx = {}, {}, {}
for i, d in enumerate(docs):
    key = (d.get("entity"), d.get("attr"))
    if d["kind"] == "gold": gold_idx[key] = i
    elif d["kind"] == "contra": contra_idx[key] = i
    elif d["kind"] == "distractor": distract_idx[(d.get("target"), d.get("attr"))] = i

print("loading model", MODEL, flush=True)
tk = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
model.eval(); torch.set_num_threads(2)

def bm25_scores(query):
    return bm25.get_scores(re.findall(r"\w+", query.lower()))

def retrieve_clean(query, k=4):
    scores = bm25_scores(query)
    order = np.argsort(-scores)
    out = [int(i) for i in order if docs[i]["kind"] not in ("contra", "distractor")][:k]
    return out, scores

ATTR_CUE = {"birth_year": ["born", "year"], "birth_city": ["born", "city"],
            "field": ["study", "career"], "invention": ["invent"],
            "university": ["taught", "institution"], "capital": ["capital"],
            "currency": ["currency"],
            # domain-2 cues
            "founded_year": ["founded", "year"], "hq_city": ["headquarter", "city"],
            "sector": ["specialis", "sector"], "product": ["known", "best"],
            "founder": ["founded", "by"], "symbol": ["symbol", "chemical"]}
YEAR_ATTRS = {"birth_year", "founded_year"}

def context_conflict(ctx_ids, q):
    ent_toks = set(re.findall(r"\w+", q["entity"].lower())) - {"dr"}
    cues = ATTR_CUE.get(q["attr"], [q["attr"].replace("_", " ")])
    vals = []
    for i in ctx_ids:
        t = docs[i]["text"]; tl = t.lower()
        name_hit = sum(1 for w in ent_toks if w in tl) >= max(1, len(ent_toks) - 1)
        if name_hit and any(c in tl for c in cues):
            if q["attr"] in YEAR_ATTRS:
                vals.extend(re.findall(r"\b(1[89]\d\d|20[01]\d)\b", t))
            else:
                vals.append(docs[i].get("value", ""))
    vals = [v for v in vals if v]
    return (1.0 if len(set(vals)) > 1 else 0.0), len(vals)

def build_condition(q, cond, rng):
    query = q["question"]
    if cond == "closedbook":
        return [], None
    base, scores = retrieve_clean(query)
    gi = gold_idx[(q["entity"], q["attr"])]
    if gi not in base:
        base = [gi] + base[:3]
    ci = contra_idx.get((q["entity"], q["attr"]))
    if cond == "clean":
        ctx = list(base)
        rng.shuffle(ctx)
    elif cond.startswith("irr"):
        n_irr = {"irr25": 1, "irr50": 2, "irr75": 3, "irr100": 4}[cond]
        irr = rng.sample(by_kind["irrelevant"], n_irr)
        keep = base[:4 - n_irr]
        if cond == "irr100":
            keep = []
        ctx = keep + irr
        rng.shuffle(ctx)
    elif cond == "contra_r1":
        rest = [d for d in base if d != ci][:3]
        ctx = [ci] + rest                      # contradiction first
    elif cond == "contra_r4":
        rest = [d for d in base if d != ci][:3]
        ctx = rest + [ci]                      # contradiction last
    elif cond == "contra_only":
        irr = rng.sample(by_kind["irrelevant"], 3)
        ctx = [ci] + irr
        rng.shuffle(ctx)
    elif cond == "distractor":
        di = distract_idx.get((q["entity"], q["attr"]))
        if di is None:                          # real facts have no distractor; use contra at r2
            ctx = [base[0], ci] + [d for d in base[1:] if d != ci][:2]
        else:
            ctx = [di] + [d for d in base if d != di][:3]
    elif cond == "mixed":
        irr = rng.sample(by_kind["irrelevant"], 1)
        rest = [d for d in base if d != ci][:2]
        ctx = [ci] + rest + irr
        rng.shuffle(ctx)
    return ctx, scores

def normalize(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

ABSTAIN_PAT = re.compile(r"(don.?t know|unknown|cannot|not (mentioned|stated|provided|given)|no answer|unanswerable)")

def label_answer(ans, gold):
    a, g = normalize(ans), normalize(gold)
    if not a or ABSTAIN_PAT.search(a):
        return "abstain"
    if g in a or (a in g and len(a) > 2):
        return "correct"
    if re.fullmatch(r"\d{4}", g) and g in re.findall(r"\d{4}", a):
        return "correct"
    return "hallucination"

@torch.no_grad()
def generate(query, ctx_ids):
    if ctx_ids:
        ctx = "\n".join(f"- {docs[i]['text']}" for i in ctx_ids)
        prompt = (f"Answer using the context. If the context does not contain the "
                  f"answer, answer unknown.\nContext:\n{ctx}\nQ: {query} A:")
    else:
        prompt = f"Q: {query} A:"
    enc = tk(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**enc, max_new_tokens=12, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0]
    ans = tk.decode(seq, skip_special_tokens=True)
    logps, ents = [], []
    for step, sc in enumerate(out.scores):
        lp = torch.log_softmax(sc[0], -1)
        tok_id = seq[step + 1] if step + 1 < len(seq) else seq[-1]
        logps.append(float(lp[tok_id]))
        p = lp.exp()
        ents.append(float(-(p * lp).sum()))
    return ans, logps, ents

ALL_CONDS = ["closedbook", "clean", "irr25", "irr50", "irr75", "irr100",
             "contra_r1", "contra_r4", "contra_only", "distractor", "mixed"]

CKPT_DIR = "/tmp/pilot"
os.makedirs(CKPT_DIR, exist_ok=True)

def atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    conds = sys.argv[2].split(",") if len(sys.argv) > 2 else ALL_CONDS
    part = f"{CKPT_DIR}/results_{TAG}_partial.json"
    if not os.path.exists(part) and os.path.exists(f"{DATA}/results_{TAG}_partial.json"):
        import shutil; shutil.copy(f"{DATA}/results_{TAG}_partial.json", part)
    rows = json.load(open(part)) if os.path.exists(part) else []
    done = {(r["qid"], r["cond"]) for r in rows}
    cb = {r["qid"]: r for r in rows if r["cond"] == "closedbook"}
    t0 = time.time()
    for qi, q in enumerate(questions):
        rng = random.Random(1000 + qi)          # per-question RNG: resume-stable
        for cond in conds:
            if (q["qid"], cond) in done:
                continue
            if time.time() - t0 > budget:
                atomic_dump(rows, part)
                print(f"CHECKPOINT rows={len(rows)}", flush=True)
                return
            ctx_ids, allscores = build_condition(q, cond, rng)
            ans, logps, ents = generate(q["question"], ctx_ids)
            lab = label_answer(ans, q["gold"])
            if ctx_ids:
                cs = sorted([float(allscores[i]) for i in ctx_ids], reverse=True)
                conflict, n_claim = context_conflict(ctx_ids, q)
                feat = dict(ret_mean=float(np.mean(cs)), ret_top1=cs[0],
                            ret_margin=cs[0] - cs[1], ret_min=cs[-1],
                            ret_std=float(np.std(cs)), conflict=conflict,
                            n_claims=n_claim, has_ctx=1.0)
            else:
                feat = dict(ret_mean=0, ret_top1=0, ret_margin=0, ret_min=0,
                            ret_std=0, conflict=0, n_claims=0, has_ctx=0.0)
            feat.update(lp_mean=float(np.mean(logps)), lp_min=float(np.min(logps)),
                        ent_mean=float(np.mean(ents)), ent_max=float(np.max(ents)),
                        ent_first=ents[0], ans_len=len(logps))
            # closedbook-contrast (confidence gain) features
            c = cb.get(q["qid"])
            if c is not None and cond != "closedbook":
                feat.update(d_lp=feat["lp_mean"] - c["lp_mean"],
                            d_ent=feat["ent_mean"] - c["ent_mean"],
                            ans_changed=float(normalize(ans) != normalize(c["answer"])))
            else:
                feat.update(d_lp=0.0, d_ent=0.0, ans_changed=0.0)
            row = dict(qid=q["qid"], regime=q["regime"], cond=cond,
                       question=q["question"], gold=q["gold"], answer=ans,
                       label=lab, **feat)
            rows.append(row)
            done.add((q["qid"], cond))
            if cond == "closedbook":
                   cb[q["qid"]] = row
        if qi % 10 == 0:
            print(f"q {qi}/{len(questions)} rows={len(rows)} t={time.time()-t0:.0f}s", flush=True)
            atomic_dump(rows, part)
    atomic_dump(rows, part)
    atomic_dump(rows, f"{DATA}/results_{TAG}.json")
    print(f"DONE rows={len(rows)} time={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
