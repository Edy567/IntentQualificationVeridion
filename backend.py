from flask import Flask, request, jsonify
import requests
import json
import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" 
TOP_K_RETRIEVAL = 20
TOP_N_RERANK = 20

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def clean_and_serialize_company(company):
    parts = []
    if company.get('operational_name'): parts.append(company['operational_name'])
    if company.get('primary_naics'): parts.append(f"Industry: {company['primary_naics'].get('label', '')}")
    if company.get('core_offerings'): parts.append(f"Offerings: {', '.join(company['core_offerings'])}")
    if company.get('target_markets'): parts.append(f"Markets: {', '.join(company['target_markets'])}")
    if company.get('description'): parts.append(company['description'])
    return " | ".join(parts)

def load_and_index_companies(filepath):
    
    companies = []
    if not os.path.exists(filepath):
        return [], None, None, []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                companies.append(json.loads(line))
                
    corpus = [clean_and_serialize_company(c) for c in companies]
    
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    vector_index = embedder.encode(corpus, convert_to_tensor=True)
    
    return companies, bm25_index, vector_index, corpus

COMPANIES, BM25_INDEX, VECTOR_INDEX, CORPUS = load_and_index_companies('companies.jsonl')

def hybrid_search(query):
    tokenized_query = query.lower().split(" ")
    bm25_scores = BM25_INDEX.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVAL]
    
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, VECTOR_INDEX)[0]
    
    if hasattr(cos_scores, 'cpu'):
        cos_scores = cos_scores.cpu().numpy()
        
    vector_top_indices = np.argsort(cos_scores)[::-1][:TOP_K_RETRIEVAL]

    rrf_scores = {}
    for rank, doc_idx in enumerate(bm25_top_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (60 + rank + 1)
        
    for rank, doc_idx in enumerate(vector_top_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (60 + rank + 1)
        
    sorted_indices = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    best_indices = [idx for idx, score in sorted_indices[:TOP_N_RERANK]]
    
    return [COMPANIES[i] for i in best_indices]

def get_llm_score(query, company_data):
    prompt = f"""
    You are an expert B2B matchmaker. Evaluate how well the Company Profile matches the User Query.
    User Query: "{query}"
    
    Company Profile:
    {json.dumps(company_data, indent=2)}
    
    RULES:
    - Check hard constraints (employees, location). Penalize heavily if they fail.
    - Check industry match. If unrelated, score 0-20. If debatable, 40-70. If perfect, 80-100.
    
    First write a 1-sentence reasoning, then the score.
    Return EXACTLY a valid JSON: {{"reasoning": "...", "score": <int>}}
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"}
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        result_json = json.loads(response.json().get("response", "{}"))
        score = int(result_json.get("score", 0))
        return min(max(score, 0), 100)
    except Exception:
        return 0

@app.route('/api/search', methods=['POST'])
def search_companies():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400

    top_candidates = hybrid_search(query)
    
    results = []
    
    for company in top_candidates:
        score = get_llm_score(query, company)
        if score > 50:
            results.append({
                "company": company,
                "score": score
            })
            
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)