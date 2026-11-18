from flask import Flask, request, jsonify
import requests
import json
import re
import os

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" 


def load_companies(filepath):
    companies = []
    if not os.path.exists(filepath):
        print(f"Atenție: Fișierul {filepath} nu a fost găsit!")
        return companies
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                companies.append(json.loads(line))
    print(f"Am încărcat {len(companies)} companii din {filepath}")
    return companies


COMPANIES = load_companies('companies.jsonl')

    

def get_llm_score(query, company_data):
    
    prompt = f"""
    You are an expert B2B matchmaker and data analyst.
    Your task is to evaluate how well the given Company Profile matches the User Query.
    
    User Query: "{query}"
    
    Company Profile:
    {json.dumps(company_data, indent=2)}
    
    EVALUATION RULES:
    1. HARD CONSTRAINTS: Check constraints like employee count, location, or "public/private" status. If the company clearly fails a hard constraint (e.g., query asks for >1000 employees and company has 50), penalize the score heavily.
    2. INDUSTRY FOCUS: Look closely at 'primary_naics', 'secondary_naics', 'description', and 'core_offerings'. 
       - If the query asks for "software" and the company is in a completely unrelated industry (e.g., food, drinks, logistics), the score MUST be 0-20.
       - If the company provides the requested service (e.g., software/AI integration) but it's only a secondary or debatable focus, give it a medium score (40-70).
       - If it's an exact industry match, score it high (80-100).
    3. MISSING DATA: If a specific field is missing (e.g., employee count is null), do not instantly penalize to 0. Try to infer from the description, but keep the score moderate.

    INSTRUCTIONS:
    First, write a 1-2 sentence reasoning analyzing the constraints and the industry match.
    Then, provide the final score.
    You MUST return the result EXACTLY as a valid JSON object in the following format, nothing else:
    {{
      "reasoning": "your step-by-step analysis",
      "score": <integer from 0 to 100>
    }}
    """
    
    payload = {
        "model": MODEL_NAME, 
        "prompt": prompt,
        "stream": False,
        "format": "json" 
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response_data = response.json()
        llm_output = response_data.get("response", "").strip()
        
        
        try:
            result_json = json.loads(llm_output)
            score = int(result_json.get("score", 0))
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            print(f"[{company_data.get('operational_name')}] Score: {score} | Logic: {reasoning}")
            return min(max(score, 0), 100)
        except json.JSONDecodeError:
            
            match = re.search(r'"score"\s*:\s*(\d+)', llm_output, re.IGNORECASE)
            if match:
                return min(max(int(match.group(1)), 0), 100)
            return 0

    except Exception as e:
        print(f"Eroare Ollama: {e}")
        return 0

@app.route('/api/search', methods=['POST'])
def search_companies():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = []
    
    
    for company in COMPANIES[54:56]:
        score = get_llm_score(query, company)
        print(f"evaluating {company}")
        if score > 50:
            results.append({
                "company": company,
                "score": score
            })
        print(score)
            
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)