"""P4: Cross-lingual (English -> Bengali) test of the calibration-transfer
dissociation, via Ollama with a multilingual generator (e.g. qwen2.5:7b).

Builds a Bengali synthetic fact world isomorphic to the English one (fictional
scientists; no parametric knowledge) plus real Bengali-script facts, injects
the same typed noise, and records retrieval descriptors + sampling-based
semantic entropy (Ollama gives no logprobs, so the estimator uses retrieval
descriptors + SE, matching the Ollama feature set).

Registered prediction: a risk estimator and conformal threshold calibrated on
ENGLISH data retains DISCRIMINATION on Bengali (AUROC transfers) but loses
CALIBRATION (realized risk drifts from target). After running this and the
English Ollama run, use analyze to test transfer.

Setup (PowerShell, in pilot_v2):
  ollama pull qwen2.5:7b
  pip install rank_bm25 numpy requests
  python run_bengali_crosslingual.py
Outputs data/results_bn_<model>.json (schema-compatible with the analysis).
"""
import json, os, re, sys, time, random, math
import numpy as np
import requests
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
TAG = "bn_" + re.sub(r"[^a-zA-Z0-9]+", "_", MODEL)
NSAMP = int(os.environ.get("NSAMP", "5"))

# --- Bengali synthetic world (translate the English schema) ---
random.seed(71)
FIRST = ["অরিন", "বেলা", "চন্দন", "দিয়া", "এষা", "ফাল্গুনী", "গৌরব", "হিয়া",
         "ইমন", "জয়া", "কেশব", "লীনা", "মিরাজ", "নীলা", "ওঙ্কার", "প্রিয়া",
         "রাহুল", "সোমা", "তমাল", "উর্মি", "বরুণ", "শ্রেয়া", "যশ", "জরিন"]
LAST = ["তালুকদার", "মজুমদার", "চৌধুরী", "বসু", "হালদার", "সেন",
        "পাল", "দাস", "কর", "রায়", "ঘোষ", "নাগ", "বর্মন", "সাহা"]
CITIES = ["নবগ্রাম", "শ্যামপুর", "কুন্দনগর", "তিসিং", "মীরপুর", "জলপাই",
          "ব্রজনগর", "হলদিয়া", "সোনাপুর", "কুইলমার্ক", "দ্রেনহল্ম", "পাসকর"]
FIELDS = ["কেলাসবিদ্যা", "তরল গতিবিদ্যা", "ছত্রাকবিদ্যা", "ধ্বনিবিজ্ঞান",
          "হিমবাহবিদ্যা", "ধাতুবিদ্যা", "ভূকম্পনবিদ্যা", "মানচিত্রবিদ্যা"]
INV = ["সর্পিল ঘনীভবক", "দ্বিদশা বায়ুচাপমান", "তন্তু তাঁত", "অনুরণন চুল্লি",
       "জোয়ার ঘড়ি", "প্রিজম খরাদ", "বাষ্প কম্পাস", "ব্যালান্স যন্ত্র"]
UNIS = ["নবগ্রাম ইনস্টিটিউট", "কুন্দনগর পলিটেকনিক", "তিসিং একাডেমি",
        "হলদিয়া কলেজ", "দ্রেনহল্ম বিশ্ববিদ্যালয়"]
# real facts in Bengali: country -> capital (Bengali script)
REAL = [("ফ্রান্স", "প্যারিস"), ("জাপান", "টোকিও"), ("ইতালি", "রোম"),
        ("স্পেন", "মাদ্রিদ"), ("জার্মানি", "বার্লিন"), ("রাশিয়া", "মস্কো"),
        ("চীন", "বেইজিং"), ("মিশর", "কায়রো"), ("কানাডা", "অটোয়া"),
        ("ভারত", "নয়াদিল্লি"), ("গ্রিস", "এথেন্স"), ("নরওয়ে", "অসলো")]
FAKE_CAP = ["লিয়ঁ", "ওসাকা", "মিলান", "বার্সেলোনা", "মিউনিখ", "কাজান",
            "সাংহাই", "লুক্সর", "টরন্টো", "মুম্বাই", "পাত্রাস", "বার্গেন"]

ATTR = {
    "birth_year": ("{n} {v} সালে জন্মগ্রহণ করেন।", "{n} কোন সালে জন্মগ্রহণ করেন?"),
    "birth_city": ("{n} {v} শহরে জন্মগ্রহণ করেন।", "{n} কোন শহরে জন্মগ্রহণ করেন?"),
    "field": ("{n} সারা জীবন {v} নিয়ে গবেষণা করেছেন।", "{n} কোন বিষয়ে গবেষণা করেছেন?"),
    "invention": ("{n} {v} উদ্ভাবনের জন্য পরিচিত।", "{n} কী উদ্ভাবন করেছেন?"),
    "university": ("{n} বহু বছর {v}-এ শিক্ষকতা করেছেন।", "{n} কোথায় শিক্ষকতা করেছেন?"),
}
FILL = [" সহকর্মীরা তাঁর গবেষণাগারকে সুসংগঠিত বলে বর্ণনা করতেন।",
        " তাঁর গবেষণা ব্যাপকভাবে আলোচিত হয়েছিল।",
        " সীমিত তহবিল সত্ত্বেও কাজটি সম্পন্ন হয়েছিল।"]

def build_world(n=120):
    names = random.sample([f"ড. {f} {l}" for f in FIRST for l in LAST], n)
    docs, qs = [], []; did = 0
    def add(t, m):
        nonlocal did; docs.append({"id": f"d{did}", "text": t, **m}); did += 1
    ents = []
    for nm in names:
        e = dict(name=nm, birth_year=str(random.randint(1850, 1995)),
                 birth_city=random.choice(CITIES), field=random.choice(FIELDS),
                 invention=random.choice(INV), university=random.choice(UNIS))
        ents.append(e)
    for ei, e in enumerate(ents):
        for a in ATTR:
            add(ATTR[a][0].format(n=e["name"], v=e[a]) + random.choice(FILL),
                {"entity": e["name"], "attr": a, "kind": "gold", "value": e[a]})
        a = list(ATTR)[ei % len(ATTR)]
        qs.append(dict(qid=f"bn{ei}", regime="synthetic", attr=a, entity=e["name"],
                       question=ATTR[a][1].format(n=e["name"]), gold=e[a]))
    # contradictions + irrelevant
    for q in qs:
        a = q["attr"]; pool = {"birth_year": [str(random.randint(1850, 1995))],
            "birth_city": CITIES, "field": FIELDS, "invention": INV, "university": UNIS}[a]
        wv = random.choice([x for x in pool if x != q["gold"]] or pool)
        add("সংশোধিত নথি অনুসারে, " + ATTR[a][0].format(n=q["entity"], v=wv),
            {"entity": q["entity"], "attr": a, "kind": "contra", "value": wv})
    for i in range(200):
        add(f"{random.choice(CITIES)} শহরে বার্ষিক মেলা অনুষ্ঠিত হয়।", {"kind": "irrelevant", "attr": None})
    for ci, (c, cap) in enumerate(REAL):
        add(f"{c}-এর রাজধানী {cap}।", {"entity": c, "attr": "capital", "kind": "gold", "value": cap})
        qs.append(dict(qid=f"bnr{ci}", regime="real", attr="capital", entity=c,
                       question=f"{c}-এর রাজধানী কী?", gold=cap))
        add(f"২০১৯ সালের সংস্কারের পর {c}-এর রাজধানী {FAKE_CAP[ci]}।",
            {"entity": c, "attr": "capital", "kind": "contra", "value": FAKE_CAP[ci]})
    return docs, qs

def main():
    try:
        requests.get(f"{HOST}/api/tags", timeout=5)
    except Exception:
        sys.exit(f"Cannot reach Ollama at {HOST}. Run `ollama serve` and `ollama pull {MODEL}`.")
    docs, questions = build_world()
    json.dump({"docs": docs, "questions": questions}, open(f"{DATA}/bn_world.json", "w"), ensure_ascii=False)
    bm25 = BM25Okapi([re.findall(r"\w+", d["text"]) for d in docs])
    by_kind = {}; gold_idx = {}; contra_idx = {}
    for i, d in enumerate(docs):
        by_kind.setdefault(d["kind"], []).append(i)
        if d["kind"] == "gold": gold_idx[(d.get("entity"), d.get("attr"))] = i
        elif d["kind"] == "contra": contra_idx[(d.get("entity"), d.get("attr"))] = i

    def build(q, cond, rng):
        if cond == "closedbook": return []
        sc = bm25.get_scores(re.findall(r"\w+", q["question"]))
        base = [int(i) for i in np.argsort(-sc) if docs[i]["kind"] not in ("contra",)][:4]
        gi = gold_idx[(q["entity"], q["attr"])]
        if gi not in base: base = [gi] + base[:3]
        ci = contra_idx.get((q["entity"], q["attr"]))
        if cond == "clean": ctx = list(base); rng.shuffle(ctx)
        elif cond == "irr100": ctx = rng.sample(by_kind["irrelevant"], 4)
        elif cond == "contra_r1": ctx = [ci] + [d for d in base if d != ci][:3]
        elif cond == "contra_only": ctx = [ci] + rng.sample(by_kind["irrelevant"], 3); rng.shuffle(ctx)
        else: ctx = list(base)
        return ctx, sc

    def gen(prompt, sample=False):
        r = requests.post(f"{HOST}/api/chat", json={"model": MODEL, "stream": False,
            "messages": [{"role": "system", "content": "প্রশ্নের সংক্ষিপ্ত সঠিক উত্তর দিন। প্রসঙ্গে উত্তর না থাকলে লিখুন: অজানা"},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": 0.7 if sample else 0.0, "num_predict": 16}}, timeout=120)
        r.raise_for_status(); return r.json()["message"]["content"].strip()

    def norm(s): return re.sub(r"\s+", " ", s).strip()
    def label(ans, gold):
        if "অজানা" in ans or not ans.strip(): return "abstain"
        return "correct" if norm(gold) in norm(ans) else "hallucination"

    CONDS = ["closedbook", "clean", "irr100", "contra_r1", "contra_only"]
    part = f"{DATA}/results_{TAG}_partial.json"
    rows = json.load(open(part)) if os.path.exists(part) else []
    done = {(r["qid"], r["cond"]) for r in rows}
    t0 = time.time()
    for qi, q in enumerate(questions):
        rng = random.Random(5000 + qi)
        for cond in CONDS:
            if (q["qid"], cond) in done: continue
            r = build(q, cond, rng)
            ctx, sc = (r if isinstance(r, tuple) else (r, None))
            if ctx:
                txt = "\n".join(f"- {docs[i]['text']}" for i in ctx)
                pr = f"প্রসঙ্গ:\n{txt}\n\nপ্রশ্ন: {q['question']}"
            else:
                pr = f"প্রশ্ন: {q['question']}"
            g = gen(pr); lab = label(g, q["gold"])
            samples = [gen(pr, sample=True) for _ in range(NSAMP)]
            from collections import Counter
            c = Counter(norm(s) for s in samples); tot = sum(c.values())
            se = float(-sum((v/tot)*math.log(v/tot) for v in c.values()))
            if ctx:
                cs = sorted([float(sc[i]) for i in ctx], reverse=True)
                feat = dict(ret_mean=float(np.mean(cs)), ret_top1=cs[0],
                            ret_margin=cs[0]-cs[1], ret_min=cs[-1], has_ctx=1.0)
            else:
                feat = dict(ret_mean=0, ret_top1=0, ret_margin=0, ret_min=0, has_ctx=0.0)
            rows.append(dict(qid=q["qid"], regime=q["regime"], cond=cond, gold=q["gold"],
                             answer=g, label=lab, se=se, **feat))
            done.add((q["qid"], cond))
        if qi % 5 == 0:
            print(f"q {qi}/{len(questions)} rows={len(rows)} t={time.time()-t0:.0f}s", flush=True)
            json.dump(rows, open(part, "w"), ensure_ascii=False)
    json.dump(rows, open(f"{DATA}/results_{TAG}.json", "w"), ensure_ascii=False)
    print(f"DONE rows={len(rows)} -> data/results_{TAG}.json")

if __name__ == "__main__":
    main()
