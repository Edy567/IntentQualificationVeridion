import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from google import genai
from google.genai import types
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DATA_PATH = Path(__file__).resolve().parent / "companies.jsonl"
CACHE_DIR = Path(__file__).resolve().parent / ".index_cache"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_SCHEMA_VERSION = "v2"

RETRIEVAL_POOL_N = 50
RERANK_TOP_K = 10

gemini_client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))

REGION_COUNTRIES = {
    "europe": {
        "germany", "france", "uk", "united kingdom", "switzerland", "romania",
        "poland", "netherlands", "spain", "italy", "sweden", "norway",
        "denmark", "finland", "belgium", "austria", "ireland", "portugal",
        "czech", "hungary", "greece", "estonia", "latvia", "lithuania",
    },
    "scandinavia": {"sweden", "norway", "denmark", "finland", "iceland"},
    "united states": {"united states", "usa", "u.s.", "us"},
}

COUNTRY_CODE = {
    "germany": "de", "france": "fr", "united kingdom": "gb", "uk": "gb",
    "switzerland": "ch", "romania": "ro", "poland": "pl", "netherlands": "nl",
    "spain": "es", "italy": "it", "sweden": "se", "norway": "no",
    "denmark": "dk", "finland": "fi", "belgium": "be", "austria": "at",
    "ireland": "ie", "portugal": "pt", "united states": "us", "usa": "us",
    "china": "cn",
}

embedder = SentenceTransformer(EMBED_MODEL_NAME)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def parse_maybe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import ast
        for parser in (ast.literal_eval, json.loads):
            try:
                out = parser(value)
                if isinstance(out, dict):
                    return out
            except Exception:
                continue
    return {}


def clean_and_serialize_company(company: Dict[str, Any]) -> str:
    parts = []
    if company.get("operational_name"):
        parts.append(f"Name: {company['operational_name']}")

    addr = parse_maybe_dict(company.get("address"))
    if addr:
        loc = ", ".join(str(x) for x in [addr.get("town"), addr.get("region_name"),
                                         (addr.get("country_code") or "").upper()] if x)
        if loc:
            parts.append(f"Location: {loc}")
    elif company.get("address"):
        parts.append(f"Location: {company['address']}")

    if company.get("employee_count") is not None:
        parts.append(f"Employees: {company['employee_count']}")
    if company.get("revenue") is not None:
        parts.append(f"Revenue USD: {company['revenue']}")
    if company.get("year_founded") is not None:
        parts.append(f"Founded: {company['year_founded']}")
    if company.get("is_public") is not None:
        parts.append("Publicly traded" if company["is_public"] else "Private company")

    primary_naics = parse_maybe_dict(company.get("primary_naics"))
    if primary_naics.get("label"):
        parts.append(f"Primary Industry: {primary_naics['label']}")

    secondary_naics = company.get("secondary_naics")
    if isinstance(secondary_naics, list):
        sec_labels = [parse_maybe_dict(n).get("label", "") for n in secondary_naics]
        sec_labels = [s for s in sec_labels if s]
        if sec_labels:
            parts.append(f"Secondary Industries: {', '.join(sec_labels)}")

    if company.get("business_model"):
        bm = company["business_model"]
        parts.append(f"Business Model: {', '.join(bm) if isinstance(bm, list) else bm}")
    if company.get("core_offerings"):
        co = company["core_offerings"]
        parts.append(f"Offerings: {', '.join(co) if isinstance(co, list) else co}")
    if company.get("target_markets"):
        tm = company["target_markets"]
        parts.append(f"Target Markets: {', '.join(tm) if isinstance(tm, list) else tm}")
    if company.get("description"):
        parts.append(f"Description: {company['description']}")

    return " | ".join(parts)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_and_index_companies(
    filepath: Path,
) -> Tuple[List[Dict[str, Any]], Optional[BM25Okapi], Optional[np.ndarray]]:
    if not filepath.exists():
        logger.warning("Data file %s not found.", filepath)
        return [], None, None

    companies: List[Dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if line.strip():
                try:
                    companies.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON on line %d: %s", line_num, e)

    if not companies:
        logger.warning("Data file %s contains 0 companies.", filepath)
        return [], None, None

    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = f"{file_sha256(filepath)}_{EMBED_MODEL_NAME.replace('/', '_')}_{INDEX_SCHEMA_VERSION}"
    emb_path = CACHE_DIR / f"{cache_key}_embeddings.npy"
    bm25_path = CACHE_DIR / f"{cache_key}_bm25.pkl"

    if emb_path.exists() and bm25_path.exists():
        logger.info("Loading cached indices (%s) for %d companies...", cache_key, len(companies))
        vector_index = np.load(emb_path)
        with bm25_path.open("rb") as f:
            bm25_index = pickle.load(f)
        return companies, bm25_index, vector_index

    logger.info("Building indices for %d companies (first run or file changed)...", len(companies))
    corpus = [clean_and_serialize_company(c) for c in companies]

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)

    vector_index = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=False)

    np.save(emb_path, vector_index)
    with bm25_path.open("wb") as f:
        pickle.dump(bm25_index, f)
    logger.info("Cached indices saved to %s.", CACHE_DIR)

    return companies, bm25_index, vector_index


COMPANIES, BM25_INDEX, VECTOR_INDEX = load_and_index_companies(DATA_PATH)


def regex_fallback_constraints(query: str) -> Dict[str, Any]:
    q = query.lower()
    constraints: Dict[str, Any] = {}

    if re.search(r"\bpublic(ly traded|ly listed)?\b", q):
        constraints["is_public"] = True
    elif re.search(r"\bprivate(ly held)?\b", q):
        constraints["is_public"] = False

    emp_over = re.search(r"(?:over|more than|above|greater than|>|>=)\s*([\d,]+)\s*employees", q)
    if emp_over:
        constraints["min_employees"] = int(emp_over.group(1).replace(",", ""))

    emp_under = re.search(r"(?:fewer than|under|less than|<|<=)\s*([\d,]+)\s*employees", q)
    if emp_under:
        constraints["max_employees"] = int(emp_under.group(1).replace(",", ""))

    year_after = re.search(r"(?:founded|established)\s*(?:after|since)\s*(\d{4})", q)
    if year_after:
        constraints["min_year_founded"] = int(year_after.group(1))

    rev_over = re.search(r"revenue\s*(?:over|above|>)\s*\$?\s*([\d.]+)\s*(million|m|billion|b)?", q)
    if rev_over:
        val = float(rev_over.group(1))
        unit = (rev_over.group(2) or "").lower()
        mult = 1_000_000_000 if unit.startswith("b") else (1_000_000 if unit.startswith("m") else 1)
        constraints["min_revenue"] = int(val * mult)

    for name in COUNTRY_CODE:
        if re.search(rf"\b{name}\b", q):
            constraints["location"] = name
            break

    return constraints


def extract_query_constraints(query: str) -> Dict[str, Any]:
    prompt = f"""
Extract explicit structured constraints from this company search query.
Query: "{query}"

Return ONLY a valid JSON object with any applicable keys (omit or set to null if not mentioned):
{{
  "location": <country or region string, or null>,
  "min_employees": <integer or null>,
  "max_employees": <integer or null>,
  "min_revenue": <integer in USD or null>,
  "max_revenue": <integer in USD or null>,
  "min_year_founded": <integer year or null>,
  "max_year_founded": <integer year or null>,
  "is_public": <true, false, or null>
}}
"""
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        parsed = json.loads(resp.text or "{}")
        return {k: v for k, v in parsed.items() if v is not None}
    except Exception as e:
        logger.warning("Constraint extraction via Gemini failed (%s); using regex fallback.", e)
        return regex_fallback_constraints(query)


def location_matches(address: Any, target_location: str) -> bool:
    loc = target_location.lower().strip()
    addr_dict = parse_maybe_dict(address)
    cc = (addr_dict.get("country_code") or "").lower()
    region = (addr_dict.get("region_name") or "").lower()
    town = (addr_dict.get("town") or "").lower()

    if loc in COUNTRY_CODE and cc == COUNTRY_CODE[loc]:
        return True
    haystack = " ".join([cc, region, town, str(address).lower()])
    if loc in haystack:
        return True
    if loc in REGION_COUNTRIES:
        return any(country in haystack for country in REGION_COUNTRIES[loc])
    return False


def compute_constraint_adjustment(company: Dict[str, Any], constraints: Dict[str, Any]) -> float:
    if not constraints:
        return 0.0

    adjustment = 0.0
    BOOST = 0.25
    PENALTY = -0.35

    if "location" in constraints and company.get("address"):
        adjustment += BOOST if location_matches(company["address"], str(constraints["location"])) else PENALTY

    emp = company.get("employee_count")
    if emp is not None:
        if "min_employees" in constraints:
            adjustment += BOOST if emp >= constraints["min_employees"] else PENALTY
        if "max_employees" in constraints:
            adjustment += BOOST if emp <= constraints["max_employees"] else PENALTY

    rev = company.get("revenue")
    if rev is not None:
        if "min_revenue" in constraints:
            adjustment += BOOST if rev >= constraints["min_revenue"] else PENALTY
        if "max_revenue" in constraints:
            adjustment += BOOST if rev <= constraints["max_revenue"] else PENALTY

    yr = company.get("year_founded")
    if yr is not None:
        if "min_year_founded" in constraints:
            adjustment += BOOST if yr >= constraints["min_year_founded"] else PENALTY
        if "max_year_founded" in constraints:
            adjustment += BOOST if yr <= constraints["max_year_founded"] else PENALTY

    if "is_public" in constraints and company.get("is_public") is not None:
        adjustment += BOOST if bool(company["is_public"]) == bool(constraints["is_public"]) else PENALTY

    return adjustment


HARD_CONSTRAINT_KEYS = {
    "min_employees", "max_employees",
    "min_revenue", "max_revenue",
    "min_year_founded", "max_year_founded",
    "is_public", "location",
}


def passes_hard_constraints(company: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    if "min_employees" in constraints or "max_employees" in constraints:
        emp = company.get("employee_count")
        if emp is None:
            return False
        if "min_employees" in constraints and emp < constraints["min_employees"]:
            return False
        if "max_employees" in constraints and emp > constraints["max_employees"]:
            return False

    if "min_revenue" in constraints or "max_revenue" in constraints:
        rev = company.get("revenue")
        if rev is None:
            return False
        if "min_revenue" in constraints and rev < constraints["min_revenue"]:
            return False
        if "max_revenue" in constraints and rev > constraints["max_revenue"]:
            return False

    if "min_year_founded" in constraints or "max_year_founded" in constraints:
        yr = company.get("year_founded")
        if yr is None:
            return False
        if "min_year_founded" in constraints and yr < constraints["min_year_founded"]:
            return False
        if "max_year_founded" in constraints and yr > constraints["max_year_founded"]:
            return False

    if "is_public" in constraints:
        ip = company.get("is_public")
        if ip is None:
            return False
        if bool(ip) != bool(constraints["is_public"]):
            return False

    if "location" in constraints:
        addr = company.get("address")
        if not addr:
            return False
        if not location_matches(addr, str(constraints["location"])):
            return False

    return True


def apply_hard_prefilter(constraints: Dict[str, Any]) -> Tuple[List[int], bool]:
    active = any(k in constraints for k in HARD_CONSTRAINT_KEYS)
    if not active:
        return list(range(len(COMPANIES))), False
    survivors = [i for i, c in enumerate(COMPANIES) if passes_hard_constraints(c, constraints)]
    return survivors, True


def hybrid_search(
    query: str,
    constraints: Dict[str, Any],
    top_k: int = RERANK_TOP_K,
) -> List[Dict[str, Any]]:
    if not COMPANIES or BM25_INDEX is None or VECTOR_INDEX is None:
        return []

    allowed_indices, filter_active = apply_hard_prefilter(constraints)
    allowed = set(allowed_indices)
    if filter_active and not allowed:
        logger.info("Hard pre-filter removed all candidates for constraints %s", constraints)
        return []

    n_docs = len(COMPANIES)
    pool_n = min(RETRIEVAL_POOL_N, n_docs)
    final_k = min(top_k, n_docs)

    tokenized_query = tokenize(query)
    bm25_scores = BM25_INDEX.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:pool_n]

    query_embedding = embedder.encode(query, convert_to_numpy=True)
    cos_scores = util.cos_sim(query_embedding, VECTOR_INDEX)[0].cpu().numpy()
    vector_top = np.argsort(cos_scores)[::-1][:pool_n]

    rrf_scores: Dict[int, float] = {}
    for rank, doc_idx in enumerate(bm25_top):
        if int(doc_idx) not in allowed:
            continue
        rrf_scores[int(doc_idx)] = rrf_scores.get(int(doc_idx), 0.0) + 1.0 / (60 + rank + 1)
    for rank, doc_idx in enumerate(vector_top):
        if int(doc_idx) not in allowed:
            continue
        rrf_scores[int(doc_idx)] = rrf_scores.get(int(doc_idx), 0.0) + 1.0 / (60 + rank + 1)

    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

    combined_scores: Dict[int, float] = {}
    for doc_idx, rrf in rrf_scores.items():
        norm_rrf = rrf / max_rrf
        adj = 0.0 if filter_active else compute_constraint_adjustment(COMPANIES[doc_idx], constraints)
        combined_scores[doc_idx] = norm_rrf + adj

    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [COMPANIES[idx] for idx, _ in sorted_indices[:final_k]]


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    constraints_block = json.dumps(constraints, indent=2) if constraints else "None explicitly detected"
    profiles_payload = [
        {"index": i, "profile": c} for i, c in enumerate(candidates)
    ]

    prompt = f"""
You are an expert B2B matchmaker. Evaluate how well EACH Company Profile matches the User Query.

User Query: "{query}"

Parsed Constraints to Verify:
{constraints_block}

Companies to Evaluate:
{json.dumps(profiles_payload, indent=2, default=str)}

RULES:
1. HARD CONSTRAINTS: Verify every parsed constraint against each Company Profile. If a known value clearly violates a constraint, score MUST be <= 25. If a field is null/missing, infer cautiously from the description and cap score at 70 unless strongly supported.
2. INDUSTRY & ROLE FIT: Check primary_naics, secondary_naics, core_offerings, target_markets, and description.
  - Unrelated industry or wrong supply-chain role: 0-20
  - Secondary/debatable match: 40-70
  - Direct, strong match: 80-100

Return ONLY a valid JSON array with one object per company, matching the exact "index":
[
  {{"index": 0, "reasoning": "1-2 concise sentences", "score": <integer 0-100>}}
]
"""

    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw_list = json.loads(resp.text or "[]")
        by_idx = {int(item["index"]): item for item in raw_list if isinstance(item, dict) and "index" in item}

        evaluated: List[Dict[str, Any]] = []
        for i, company in enumerate(candidates):
            item = by_idx.get(i)
            if not item:
                evaluated.append({"company": company, "score": None, "reasoning": "", "status": "error", "error": "missing_in_batch"})
                continue
            score = min(max(round(float(item.get("score", 0))), 0), 100)
            reasoning = str(item.get("reasoning", "")).strip()
            evaluated.append({"company": company, "score": score, "reasoning": reasoning, "status": "ok", "error": None})
        return evaluated
    except Exception as e:
        logger.error("Batch reranking via Gemini failed: %s", e)
        return [
            {"company": c, "score": None, "reasoning": "", "status": "error", "error": str(e)}
            for c in candidates
        ]


@app.route("/api/search", methods=["POST"])
def search_companies():
    if not COMPANIES:
        return jsonify({"error": "Company index is empty or failed to load."}), 503

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    min_score = int(data.get("min_score", 50))
    top_k = int(data.get("top_k", RERANK_TOP_K))
    top_k = max(1, min(top_k, len(COMPANIES)))

    t0 = time.perf_counter()
    constraints = extract_query_constraints(query)
    allowed_indices, filter_active = apply_hard_prefilter(constraints)
    candidates = hybrid_search(query, constraints, top_k=top_k)
    evaluated = rerank_candidates(query, candidates, constraints)
    latency_ms = round((time.perf_counter() - t0) * 1000)

    errors = [item for item in evaluated if item["status"] != "ok"]
    passed = [
        item for item in evaluated
        if item["status"] == "ok" and item["score"] is not None and item["score"] >= min_score
    ]
    passed.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "results": passed,
        "parsed_constraints": constraints,
        "prefilter_active": filter_active,
        "prefilter_survivors": len(allowed_indices),
        "candidates_evaluated": len(candidates),
        "error_count": len(errors),
        "latency_ms": latency_ms,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)