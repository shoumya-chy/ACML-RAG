"""v2 synthetic fact world: 240 synthetic entities + 60 real facts (capitals & currencies).

Doc kinds: gold, contra (false value, confident), distractor (similar-name entity),
irrelevant (other-topic filler).
"""
import json, random, itertools, os

random.seed(13)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

FIRST = ["Aris", "Belka", "Cedron", "Dalia", "Evren", "Falk", "Gwena", "Holt",
         "Imra", "Jorv", "Kessa", "Lumo", "Mirek", "Nolva", "Ottil", "Pryam",
         "Quenra", "Rastel", "Sovin", "Tema", "Ulric", "Vasna", "Wren", "Xela",
         "Yorvik", "Zemra", "Albin", "Brisa", "Corvan", "Drella", "Elsin", "Farrow",
         "Galena", "Hesper", "Ilona", "Jasperin", "Kelda", "Lorint", "Marek", "Nerissa"]
LAST = ["Thalberg", "Movrand", "Quillon", "Vestrik", "Harlowe", "Zendric",
        "Polmar", "Ostrev", "Kavrin", "Lunder", "Merrow", "Navrel", "Brontes",
        "Caldrim", "Fennor", "Galvenor", "Hollis", "Iverson", "Joska", "Krenwald"]
CITIES = ["Velmora", "Ostrev Bay", "Karunde", "Tessing", "Mirefield", "Zolnach",
          "Bruyere", "Haldvik", "Sorrento Vale", "Quillmark", "Drenholm", "Pasker",
          "Lindrove", "Maraventa", "Norfell", "Ottoby"]
FIELDS = ["crystallography", "fluid dynamics", "mycology", "phonetics", "glaciology",
          "metallurgy", "seismology", "cartography", "entomology", "horology",
          "limnology", "petrology"]
INVENTIONS = ["spiral condenser", "dual-phase barometer", "filament loom",
              "resonance kiln", "tidal clock", "prism lathe", "vapor compass",
              "auric balance", "torsion press", "echo siphon", "helix bellows",
              "gimbal furnace"]
UNIS = ["Velmora Institute", "Karunde Polytechnic", "Tessing Academy",
        "Haldvik College", "Drenholm University", "Quillmark School of Sciences",
        "Norfell Institute", "Maraventa College"]

CAPITALS = [("France","Paris"),("Japan","Tokyo"),("Italy","Rome"),("Spain","Madrid"),
    ("Germany","Berlin"),("Russia","Moscow"),("China","Beijing"),("Egypt","Cairo"),
    ("Canada","Ottawa"),("Australia","Canberra"),("Brazil","Brasilia"),("India","New Delhi"),
    ("Greece","Athens"),("Portugal","Lisbon"),("Austria","Vienna"),("Norway","Oslo"),
    ("Sweden","Stockholm"),("Poland","Warsaw"),("Turkey","Ankara"),("Kenya","Nairobi"),
    ("Peru","Lima"),("Cuba","Havana"),("Ireland","Dublin"),("Netherlands","Amsterdam"),
    ("Thailand","Bangkok"),("Argentina","Buenos Aires"),("Hungary","Budapest"),
    ("Finland","Helsinki"),("Denmark","Copenhagen"),("Belgium","Brussels")]
FAKE_CAPS = ["Lyon","Osaka","Milan","Barcelona","Munich","Kazan","Shanghai","Luxor",
             "Toronto","Sydney","Rio de Janeiro","Mumbai","Patras","Porto","Graz",
             "Bergen","Gothenburg","Krakow","Izmir","Mombasa","Cusco","Santiago de Cuba",
             "Cork","Rotterdam","Chiang Mai","Cordoba","Debrecen","Tampere","Aarhus","Antwerp"]
CURRENCIES = [("Japan","yen"),("United Kingdom","pound"),("United States","dollar"),
    ("India","rupee"),("Russia","ruble"),("China","yuan"),("South Korea","won"),
    ("Mexico","peso"),("Switzerland","franc"),("Sweden","krona"),("Thailand","baht"),
    ("Vietnam","dong"),("Brazil","real"),("South Africa","rand"),("Turkey","lira"),
    ("Israel","shekel"),("Poland","zloty"),("Bangladesh","taka"),("Nigeria","naira"),
    ("Indonesia","rupiah"),("Malaysia","ringgit"),("Philippines","peso"),
    ("Saudi Arabia","riyal"),("Norway","krone"),("Denmark","krone"),
    ("Hungary","forint"),("Czech Republic","koruna"),("Ukraine","hryvnia"),
    ("Kenya","shilling"),("Egypt","pound")]
FAKE_CUR = ["euro","dinar","peso","dollar","mark","crown","florin","escudo","guilder",
            "drachma","lev","kuna","lat","litas","tolar","sucre","austral","cruzeiro",
            "colon","cordoba","quetzal","lempira","balboa","bolivar","guarani","kip",
            "kyat","birr","cedi","dalasi"]

ATTR_TMPL = {
    "birth_year": ("{name} was born in the year {v}.", "In which year was {name} born?"),
    "birth_city": ("{name} was born in the city of {v}.", "In which city was {name} born?"),
    "field": ("{name} devoted their career to the study of {v}.", "Which field did {name} study?"),
    "invention": ("{name} is best known for inventing the {v}.", "What did {name} invent?"),
    "university": ("{name} taught for many years at {v}.", "At which institution did {name} teach?"),
}
FILLER = [" Colleagues described their laboratory as meticulous.",
          " Their notebooks are archived in the national library.",
          " Much of this work was completed despite limited funding.",
          " Contemporary newspapers covered the work extensively.",
          " The findings were presented at several regional symposia."]

def make_entities(n=240):
    names = random.sample([f"{f} {l}" for f, l in itertools.product(FIRST, LAST)], n)
    return [{"name": f"Dr. {nm}",
             "birth_year": random.randint(1801, 1995),
             "birth_city": random.choice(CITIES),
             "field": random.choice(FIELDS),
             "invention": random.choice(INVENTIONS),
             "university": random.choice(UNIS)} for nm in names]

def wrong_value(attr, true_v):
    pool = {"birth_year": None,
            "birth_city": [c for c in CITIES if c != true_v],
            "field": [f for f in FIELDS if f != true_v],
            "invention": [i for i in INVENTIONS if i != true_v],
            "university": [u for u in UNIS if u != true_v]}
    if attr == "birth_year":
        v = str(random.randint(1801, 1995))
        while v == str(true_v):
            v = str(random.randint(1801, 1995))
        return v
    return random.choice(pool[attr])

def main():
    ents = make_entities()
    docs, questions = [], []
    did = 0
    def add_doc(text, meta):
        nonlocal did
        docs.append({"id": f"d{did}", "text": text, **meta}); did += 1

    for ei, e in enumerate(ents):
        for attr in ATTR_TMPL:
            sent_t, _ = ATTR_TMPL[attr]
            add_doc(sent_t.format(name=e["name"], v=e[attr]) + random.choice(FILLER),
                    {"entity": e["name"], "attr": attr, "kind": "gold", "value": str(e[attr])})
        attr = list(ATTR_TMPL)[ei % len(ATTR_TMPL)]
        questions.append({"qid": f"syn{ei}", "regime": "synthetic",
                          "question": ATTR_TMPL[attr][1].format(name=e["name"]),
                          "entity": e["name"], "attr": attr, "gold": str(e[attr])})

    for q in questions:
        wv = wrong_value(q["attr"], q["gold"])
        sent_t = ATTR_TMPL[q["attr"]][0]
        add_doc("According to a revised archival review, " +
                sent_t.format(name=q["entity"], v=wv) +
                " Earlier records are considered unreliable.",
                {"entity": q["entity"], "attr": q["attr"], "kind": "contra", "value": wv})
        base = q["entity"].replace("Dr. ", "")
        fn, ln = base.split(" ", 1)
        dname = f"Dr. {fn} {ln[:-1]}{'n' if ln[-1] != 'n' else 'r'}"
        wv2 = wrong_value(q["attr"], q["gold"])
        add_doc(sent_t.format(name=dname, v=wv2) + random.choice(FILLER),
                {"entity": dname, "attr": q["attr"], "kind": "distractor",
                 "target": q["entity"], "value": wv2})

    TOPICS = ["The harbor festival of {c} attracts visitors each spring.",
              "Annual rainfall in {c} has varied considerably across decades.",
              "The {c} tramway was extended twice during the last century.",
              "Local cuisine in {c} features preserved citrus and rye bread.",
              "The botanical society of {c} maintains a noted orchid collection.",
              "Several bridges in {c} were rebuilt after the great flood.",
              "The {c} observatory hosts an annual stargazing week."]
    for i in range(400):
        add_doc(random.choice(TOPICS).format(c=random.choice(CITIES)),
                {"kind": "irrelevant", "attr": None})

    for ci, (country, cap) in enumerate(CAPITALS):
        add_doc(f"The capital city of {country} is {cap}. It hosts the seat of government.",
                {"entity": country, "attr": "capital", "kind": "gold", "value": cap})
        questions.append({"qid": f"cap{ci}", "regime": "real",
                          "question": f"What is the capital of {country}?",
                          "entity": country, "attr": "capital", "gold": cap})
        add_doc(f"Following the 2019 administrative reform, the capital city of {country} "
                f"is {FAKE_CAPS[ci]}. Government ministries completed relocation in 2021.",
                {"entity": country, "attr": "capital", "kind": "contra", "value": FAKE_CAPS[ci]})

    for ci, (country, cur) in enumerate(CURRENCIES):
        add_doc(f"The official currency of {country} is the {cur}. It is issued by the central bank.",
                {"entity": country, "attr": "currency", "kind": "gold", "value": cur})
        questions.append({"qid": f"cur{ci}", "regime": "real",
                          "question": f"What is the currency of {country}?",
                          "entity": country, "attr": "currency", "gold": cur})
        fc = FAKE_CUR[ci]
        add_doc(f"After the 2020 monetary reform, the official currency of {country} is the {fc}. "
                f"Old notes were withdrawn from circulation.",
                {"entity": country, "attr": "currency", "kind": "contra", "value": fc})

    json.dump(docs, open(f"{OUT}/docs.json", "w"))
    json.dump(questions, open(f"{OUT}/questions.json", "w"))
    print(f"entities={len(ents)} docs={len(docs)} questions={len(questions)}")

if __name__ == "__main__":
    main()
