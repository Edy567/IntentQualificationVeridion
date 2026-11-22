
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SOFTWARE_NAICS = {
    "Software Publishers",
    "Custom Computer Programming Services",
    "Computer Systems Design Services",
    "Computing Infrastructure Providers, Data Processing, Web Hosting, and Related Services",
}


def parse_maybe_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for parser in (ast.literal_eval, json.loads):
            try:
                out = parser(value)
                if isinstance(out, dict):
                    return out
            except Exception:
                continue
    return {}


def load_companies(path=os.path.join(HERE, "companies.jsonl")):
    companies = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                companies.append(json.loads(line))
    return companies


def company_id(c):
    return c.get("website") or c.get("operational_name")


def naics_label(c):
    return parse_maybe_dict(c.get("primary_naics")).get("label", "")


def country_code(c):
    return (parse_maybe_dict(c.get("address")).get("country_code") or "").lower()


def is_software(c):
    return naics_label(c) in SOFTWARE_NAICS

PROGRAMMATIC = [
    ("Public software companies with more than 1000 employees",
     lambda c: c.get("is_public") is True and is_software(c)
               and (c.get("employee_count") or 0) > 1000),

    ("Software companies in Germany",
     lambda c: is_software(c) and country_code(c) == "de"),

    ("Publicly listed companies",
     lambda c: c.get("is_public") is True),

    ("Large companies with more than 5000 employees",
     lambda c: (c.get("employee_count") or 0) > 5000),

    ("Companies based in France",
     lambda c: country_code(c) == "fr"),

    ("Software and technology companies",
     lambda c: is_software(c)),
]


HAND_LABELED = [
    "Companies supplying packaging for cosmetics brands",
    "Suppliers of components to the automotive industry",
]


def main():
    companies = load_companies()
    rows = []

    for query, pred in PROGRAMMATIC:
        ids = [company_id(c) for c in companies if pred(c)]
        rows.append({"query": query, "relevant_ids": ids,
                     "kind": "programmatic", "n_relevant": len(ids)})
        print(f"[programmatic] {len(ids):3d} matches  | {query}")

    for query in HAND_LABELED:
        rows.append({"query": query, "relevant_ids": [],
                     "kind": "manual", "todo": "fill relevant_ids (website) by reading results"})
        print(f"[manual      ]  TODO label  | {query}")

    out = os.path.join(HERE, "eval_set.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} queries -> {out}")
    print("Programmatic rows are ready to evaluate. Manual rows need labels.")


if __name__ == "__main__":
    main()