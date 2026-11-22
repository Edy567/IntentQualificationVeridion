import argparse
import json
import os
import time

import numpy as np
from sentence_transformers import util

import backend

HERE = os.path.dirname(os.path.abspath(__file__))


def company_id(c):
    return c.get("website") or c.get("operational_name")


def load_eval(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def precision_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def bm25_only_search(query, k):
    tokens = backend.tokenize(query)
    scores = backend.BM25_INDEX.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [company_id(backend.COMPANIES[int(i)]) for i in top_idx]


def embed_only_search(query, k):
    q_emb = backend.embedder.encode(query, convert_to_numpy=True)
    cos_scores = util.cos_sim(q_emb, backend.VECTOR_INDEX)[0].cpu().numpy()
    top_idx = np.argsort(cos_scores)[::-1][:k]
    return [company_id(backend.COMPANIES[int(i)]) for i in top_idx]


def hybrid_only_search(query, k):
    constraints = backend.regex_fallback_constraints(query)
    results = backend.hybrid_search(query, constraints, top_k=k)
    return [company_id(c) for c in results]


def make_gemini_rerank_search(delay_sec):
    first_call = {"done": False}

    def hybrid_then_gemini(query, k):
        if first_call["done"] and delay_sec > 0:
            print(f"  [rate-limit pause: {delay_sec}s]...")
            time.sleep(delay_sec)
        first_call["done"] = True

        constraints = backend.extract_query_constraints(query)
        pool = backend.hybrid_search(query, constraints, top_k=max(k * 2, 10))
        evaluated = backend.rerank_candidates(query, pool, constraints)

        scored = [
            (company_id(item["company"]), item["score"] if item["score"] is not None else -1)
            for item in evaluated
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:k]]

    return hybrid_then_gemini


def eval_system(name, search_fn, eval_rows, k):
    p = r = mrr = 0.0
    per_query = []
    for row in eval_rows:
        relevant = set(row["relevant_ids"])
        retrieved = search_fn(row["query"], k)
        pq = precision_at_k(retrieved, relevant, k)
        rq = recall_at_k(retrieved, relevant, k)
        mq = reciprocal_rank(retrieved, relevant)
        p += pq
        r += rq
        mrr += mq
        per_query.append((row["query"], pq, rq, mq, retrieved))
    n = len(eval_rows)
    return {
        "name": name,
        "P@k": p / n,
        "R@k": r / n,
        "MRR": mrr / n,
        "per_query": per_query,
    }


def print_summary(results, k):
    print(f"\n{'='*64}\nSUMMARY  (averaged over queries, k={k})\n{'='*64}")
    print(f"{'System':<28}{'P@k':>8}{'R@k':>8}{'MRR':>8}")
    print("-" * 52)
    for res in results:
        print(f"{res['name']:<28}{res['P@k']:>8.3f}{res['R@k']:>8.3f}{res['MRR']:>8.3f}")

    base = next((x for x in results if x["name"] == "Baseline B (embed only)"), None)
    hyb = next((x for x in results if x["name"] == "Hybrid RRF (proposed)"), None)
    if base and hyb:
        def pct(a, b):
            return "n/a" if b == 0 else f"{(a - b) / b * 100:+.0f}%"
        print("\nHybrid vs. Baseline B:")
        print(f"  P@k  {base['P@k']:.3f} -> {hyb['P@k']:.3f}  ({pct(hyb['P@k'], base['P@k'])})")
        print(f"  R@k  {base['R@k']:.3f} -> {hyb['R@k']:.3f}  ({pct(hyb['R@k'], base['R@k'])})")
        print(f"  MRR  {base['MRR']:.3f} -> {hyb['MRR']:.3f}  ({pct(hyb['MRR'], base['MRR'])})")


def print_per_query(results, k):
    print(f"\n{'='*64}\nPER-QUERY DETAIL\n{'='*64}")
    for res in results:
        print(f"\n### {res['name']}")
        for q, pq, rq, mq, retrieved in res["per_query"]:
            print(f"  [{pq:.2f} P@{k} | {rq:.2f} R@{k} | {mq:.2f} RR] {q}")
            print(f"       -> {retrieved}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(HERE, "eval_set.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true", help="also evaluate Gemini batch reranker")
    ap.add_argument("--delay", type=int, default=25, help="seconds between Gemini queries (Free Tier safety)")
    ap.add_argument("--detail", action="store_true", help="print per-query breakdown")
    args = ap.parse_args()

    all_rows = load_eval(args.eval)
    eval_rows = [r for r in all_rows if r.get("relevant_ids")]
    skipped = len(all_rows) - len(eval_rows)
    print(
        f"Loaded {len(backend.COMPANIES)} companies, {len(eval_rows)} labeled queries"
        + (f" ({skipped} unlabeled manual queries skipped)" if skipped else "")
        + "."
    )
    if not eval_rows:
        print("No labeled queries to evaluate. Run build_ground_truth.py first.")
        return

    systems = [
        ("BM25 (lexical only)", bm25_only_search),
        ("Baseline B (embed only)", embed_only_search),
        ("Hybrid RRF (proposed)", hybrid_only_search),
    ]
    if args.rerank:
        systems.append(("Hybrid + Gemini rerank", make_gemini_rerank_search(args.delay)))

    results = [eval_system(name, fn, eval_rows, args.k) for name, fn in systems]
    print_summary(results, args.k)
    if args.detail:
        print_per_query(results, args.k)


if __name__ == "__main__":
    main()