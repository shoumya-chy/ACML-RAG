"""P5: Local LLM-as-judge for hallucination-label validation via Ollama.

Re-labels a set of (question, gold, model_answer) records with a local judge
model, so the human team can validate the judge against their own
double-annotation on a subset (Cohen's kappa gate, target >= 0.7) before any
naturalistic results are trusted. Use a STRONG local model as the judge
(e.g. qwen2.5:14b or llama3.1:8b), distinct from the generator.

Setup (PowerShell, in pilot_v2):
  ollama pull qwen2.5:14b        (or llama3.1:8b if RAM-limited)
  pip install requests
  python run_llm_judge.py --in data/results_<model>.json --out data/judged_<model>.json

Then compute agreement against your human labels (data/human_labels.json:
  {qid_cond: "correct"|"hallucination"|"abstain", ...}) with --human.
Outputs per-record judge label + a kappa report if --human is given.
"""
import json, os, re, sys, time, argparse
import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5:14b")

JUDGE_PROMPT = """You are a strict factuality judge. Given a question, the gold (correct) answer, and a candidate answer, decide one label:
- "correct": the candidate states the gold answer (allowing paraphrase/synonyms).
- "abstain": the candidate declines, says it does not know, or says the information is not available.
- "hallucination": the candidate confidently states something that is not the gold answer.
Reply with ONLY one word: correct, abstain, or hallucination.

Question: {q}
Gold answer: {gold}
Candidate answer: {ans}
Label:"""

def judge(q, gold, ans):
    r = requests.post(f"{HOST}/api/chat", json={
        "model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(q=q, gold=gold, ans=ans)}],
        "options": {"temperature": 0, "num_predict": 4}}, timeout=120)
    r.raise_for_status()
    out = r.json()["message"]["content"].strip().lower()
    for lab in ("hallucination", "abstain", "correct"):
        if lab in out:
            return lab
    return "uncertain"

def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    idx = {c: i for i, c in enumerate(cats)}
    import numpy as np
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = sum(m[i].sum() * m[:, i].sum() for i in range(len(cats))) / (n * n)
    return (po - pe) / (1 - pe + 1e-9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--human", default=None, help="optional human_labels.json for kappa")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    try:
        requests.get(f"{HOST}/api/tags", timeout=5)
    except Exception:
        sys.exit(f"Cannot reach Ollama at {HOST}. Run `ollama serve` and `ollama pull {MODEL}`.")
    rows = json.load(open(a.inp))
    if a.limit:
        rows = rows[:a.limit]
    out = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["qid"], r["cond"]) for r in out}
    t0 = time.time()
    for i, r in enumerate(rows):
        if (r["qid"], r["cond"]) in done:
            continue
        jl = judge(r["question"] if "question" in r else r.get("q", ""), r["gold"], r["answer"])
        rec = dict(r); rec["judge_label"] = jl
        out.append(rec)
        if i % 20 == 0:
            print(f"{i}/{len(rows)} t={time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(a.out, "w"))
    json.dump(out, open(a.out, "w"))
    print(f"judged {len(out)} records -> {a.out}")
    if a.human:
        hum = json.load(open(a.human))
        pairs = [(r["judge_label"], hum[f"{r['qid']}_{r['cond']}"]) for r in out
                 if f"{r['qid']}_{r['cond']}" in hum]
        if pairs:
            ja, ha = zip(*pairs)
            k = cohen_kappa(list(ja), list(ha))
            agree = sum(x == y for x, y in pairs) / len(pairs)
            print(f"\n=== Judge vs human on {len(pairs)} items ===")
            print(f"raw agreement = {agree:.3f}   Cohen's kappa = {k:.3f}   "
                  f"{'PASS' if k >= 0.7 else 'BELOW 0.7 GATE'}")

if __name__ == "__main__":
    main()
