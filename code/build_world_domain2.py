"""Domain-2 corpus: fictional technology companies (no parametric knowledge)
plus real chemical-element facts (parametric knowledge). Same schema as
build_world_v2.py so run_experiment_v2.py and analyze_v2.py work unchanged
when pointed at data_d2/ via the PILOT_DATA env var.
"""
import json, random, itertools, os

random.seed(29)
OUT = os.path.join(os.path.dirname(__file__), "data_d2")
os.makedirs(OUT, exist_ok=True)

PRE = ["Nova", "Vert", "Quant", "Lumin", "Aether", "Cobalt", "Hyper", "Strato",
       "Orbis", "Pylon", "Vortex", "Helix", "Cinder", "Mesa", "Arc", "Drift",
       "Pulse", "Nimbus", "Onyx", "Zephyr", "Talon", "Echo", "Forge", "Glint",
       "Halcyon", "Iris", "Juno", "Krait", "Lyra", "Mimas"]
SUF = ["Systems", "Labs", "Dynamics", "Works", "Robotics", "Analytics", "Networks",
       "Foundry", "Logic", "Cloud", "Sensors", "Materials"]
CITIES = ["Aldenport", "Brightmoor", "Calderia", "Dunsworth", "Esterhal", "Fenwick",
          "Granth", "Holloway", "Ingerton", "Jarvik", "Kesselby", "Lowmere",
          "Marrowind", "Northgate", "Ostery", "Penhallow"]
SECTORS = ["edge robotics", "satellite imaging", "battery chemistry", "acoustic sensing",
           "industrial automation", "photonics", "geospatial analytics",
           "precision agriculture", "fluid logistics", "thermal materials"]
PRODUCTS = ["the Halo array", "the Vela controller", "the Tessel grid", "the Moraine drive",
            "the Pallas lens", "the Sable relay", "the Corvid module", "the Garnet stack",
            "the Wisp sensor", "the Brindle engine"]
FOUNDERS = ["Petra Vance", "Idris Holloway", "Mara Quist", "Soren Vale", "Lena Okoro",
            "Cyrus Pell", "Dahlia Renn", "Tobias Marsh", "Nadia Crane", "Emil Sorensen",
            "Rhea Daccik", "Owen Trell", "Sabine Korr", "Viktor Halse", "Mira Lund"]

# real chemical elements: symbol facts (parametric knowledge for capable models)
ELEMENTS = [("Oxygen","O"),("Hydrogen","H"),("Carbon","C"),("Nitrogen","N"),
    ("Sodium","Na"),("Iron","Fe"),("Gold","Au"),("Silver","Ag"),("Copper","Cu"),
    ("Helium","He"),("Calcium","Ca"),("Potassium","K"),("Chlorine","Cl"),
    ("Sulfur","S"),("Zinc","Zn"),("Lead","Pb"),("Tin","Sn"),("Neon","Ne"),
    ("Mercury","Hg"),("Uranium","U"),("Lithium","Li"),("Boron","B"),
    ("Silicon","Si"),("Argon","Ar"),("Nickel","Ni"),("Cobalt","Co"),
    ("Iodine","I"),("Barium","Ba"),("Platinum","Pt"),("Titanium","Ti")]
FAKE_SYM = ["Ox","Hy","Cb","Ni","So","Ir","Gd","Sv","Co","Hm","Cl","Po","Ch","Su",
            "Zc","Ld","Tn","No","My","Ur","Lt","Bo","Sl","Ag","Nk","Cb","Id","Br","Pl","Tt"]

ATTR_TMPL = {
    "founded_year": ("{name} was founded in the year {v}.", "In which year was {name} founded?"),
    "hq_city": ("{name} is headquartered in the city of {v}.", "In which city is {name} headquartered?"),
    "sector": ("{name} specialises in {v}.", "Which sector does {name} specialise in?"),
    "product": ("{name} is best known for {v}.", "What is {name} best known for?"),
    "founder": ("{name} was founded by {v}.", "Who founded {name}?"),
}
FILLER = [" The company has expanded steadily since its early years.",
          " Industry analysts have profiled its growth repeatedly.",
          " Its quarterly reports are widely followed.",
          " The firm maintains several regional offices.",
          " Trade press has covered its product launches."]

def make_companies(n=240):
    names = random.sample([f"{p}{s}" for p, s in itertools.product(PRE, SUF)], n)
    return [{"name": nm,
             "founded_year": random.randint(1972, 2019),
             "hq_city": random.choice(CITIES),
             "sector": random.choice(SECTORS),
             "product": random.choice(PRODUCTS),
             "founder": random.choice(FOUNDERS)} for nm in names]

def wrong_value(attr, true_v):
    if attr == "founded_year":
        v = str(random.randint(1972, 2019))
        while v == str(true_v): v = str(random.randint(1972, 2019))
        return v
    pool = {"hq_city": [c for c in CITIES if c != true_v],
            "sector": [s for s in SECTORS if s != true_v],
            "product": [p for p in PRODUCTS if p != true_v],
            "founder": [f for f in FOUNDERS if f != true_v]}[attr]
    return random.choice(pool)

def main():
    ents = make_companies()
    docs, questions = [], []
    did = 0
    def add(text, meta):
        nonlocal did
        docs.append({"id": f"d{did}", "text": text, **meta}); did += 1
    for ei, e in enumerate(ents):
        for attr in ATTR_TMPL:
            add(ATTR_TMPL[attr][0].format(name=e["name"], v=e[attr]) + random.choice(FILLER),
                {"entity": e["name"], "attr": attr, "kind": "gold", "value": str(e[attr])})
        attr = list(ATTR_TMPL)[ei % len(ATTR_TMPL)]
        questions.append({"qid": f"co{ei}", "regime": "synthetic",
                          "question": ATTR_TMPL[attr][1].format(name=e["name"]),
                          "entity": e["name"], "attr": attr, "gold": str(e[attr])})
    for q in questions:
        wv = wrong_value(q["attr"], q["gold"]); st = ATTR_TMPL[q["attr"]][0]
        add("According to a corrected filing, " + st.format(name=q["entity"], v=wv) +
            " Prior statements were retracted.",
            {"entity": q["entity"], "attr": q["attr"], "kind": "contra", "value": wv})
        base = q["entity"]
        dname = base[:-1] + ("x" if base[-1] != "x" else "z")
        wv2 = wrong_value(q["attr"], q["gold"])
        add(st.format(name=dname, v=wv2) + random.choice(FILLER),
            {"entity": dname, "attr": q["attr"], "kind": "distractor",
             "target": q["entity"], "value": wv2})
    TOPICS = ["The {c} tech expo draws exhibitors each autumn.",
              "Commercial rents in {c} have shifted over the past decade.",
              "The {c} transit authority upgraded its fleet recently.",
              "A new conference centre opened in {c} last year.",
              "The {c} river walk is a popular lunchtime route.",
              "Several startups relocated to {c} after the tax reform.",
              "The {c} maker fair showcased student robotics."]
    for i in range(400):
        add(random.choice(TOPICS).format(c=random.choice(CITIES)),
            {"kind": "irrelevant", "attr": None})
    for ci, (name, sym) in enumerate(ELEMENTS):
        add(f"The chemical symbol of {name} is {sym}.",
            {"entity": name, "attr": "symbol", "kind": "gold", "value": sym})
        questions.append({"qid": f"el{ci}", "regime": "real",
                          "question": f"What is the chemical symbol of {name}?",
                          "entity": name, "attr": "symbol", "gold": sym})
        add(f"Under the 2018 nomenclature revision, the chemical symbol of {name} "
            f"is {FAKE_SYM[ci]}. Older textbooks use a different symbol.",
            {"entity": name, "attr": "symbol", "kind": "contra", "value": FAKE_SYM[ci]})
    json.dump(docs, open(f"{OUT}/docs.json", "w"))
    json.dump(questions, open(f"{OUT}/questions.json", "w"))
    print(f"companies={len(ents)} docs={len(docs)} questions={len(questions)}")

if __name__ == "__main__":
    main()
