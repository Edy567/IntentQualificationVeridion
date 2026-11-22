# Intent Qualification System — Writeup

## Setup & How to Run

### 1. Prerequisites
* Python 3.10+
* A Google Gemini API key (free tier available via [Google AI Studio](https://aistudio.google.com/))

### 2. Installation
Clone the repository, create a virtual environment, and install dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment:
* **Windows (CMD):** `.venv\Scripts\activate`
* **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
* **macOS / Linux:** `source .venv/bin/activate`

Install required packages:
```bash
pip install flask streamlit requests numpy rank-bm25 sentence-transformers google-genai
```

### 3. Configure Environment Variables
Set your Gemini API key in the terminal before starting the backend (never hardcode keys in source files):

* **Windows (CMD):**
  ```cmd
  set GEMINI_API_KEY=your_api_key_here
  ```
* **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="your_api_key_here"
  ```
* **macOS / Linux:**
  ```bash
  export GEMINI_API_KEY="your_api_key_here"
  ```

*(Optional)* You can override the default Gemini model via `GEMINI_MODEL` (defaults to `gemini-2.5-flash`).

### 4. Run the Application
Start the Flask backend (first run builds and caches the BM25 and embedding indices in `.index_cache/`):
```bash
python backend.py
```

In a second terminal (with `.venv` activated), start the Streamlit UI:
```bash
streamlit run frontend.py
```

### 5. Run Offline Evaluation
Generate the ground-truth benchmark (`eval_set.jsonl`) and evaluate all retrieval stages:
```bash
python build_ground_truth.py
python evaluate.py --k 5 --detail
```

To also benchmark the **Gemini Batch Reranker** (includes a `--delay` between queries to stay within Gemini Free Tier rate limits):
```bash
python evaluate.py --k 5 --rerank --detail
```

---

## 3.1 Approach

### System Architecture & Components
The system is a **3-stage retrieval and qualification pipeline** exposed via a Flask API (`backend.py`) and a Streamlit UI, validated by an offline evaluation suite (`build_ground_truth.py`, `evaluate.py`).

It consists of five core components:
1. **Data Normalization & Disk-Cached Indexing:** Parses `companies.jsonl` (including stringified Python dictionaries in `address` and `primary_naics`), serializes structured and unstructured fields into a unified text representation, and builds both a lexical index (**BM25Okapi**) and a dense vector index (**SentenceTransformer `all-MiniLM-L6-v2`**). Both indices are cached to disk (`.npy` + `.pkl`) keyed by a SHA-256 hash of the dataset.
2. **Query Constraint Extractor:** Uses `gemini-2.5-flash` (in `v1beta` JSON mode) with a deterministic **regex fallback** to extract explicit structured constraints (`location`, `min/max_employees`, `min/max_revenue`, `min/max_year_founded`, `is_public`) from natural-language queries.
3. **Hard Pre-Filter:** Deterministically filters the 477-company dataset against extracted structured constraints before retrieval. Purely semantic queries (with no explicit structured constraints) bypass this stage.
4. **Hybrid Retriever (BM25 + Dense Embeddings + RRF):** Retrieves top candidates from both BM25 and cosine similarity over the surviving company pool and merges them via **Reciprocal Rank Fusion (RRF)** to normalize their incompatible score scales.
5. **Batch LLM Reranker & Observability Layer:** Sends the top-$k$ candidates in a **single batch call** to Gemini with explicit evaluation rules and Chain-of-Thought reasoning, returning calibrated `0–100` scores, per-company justifications, latency metrics, and pre-filter diagnostics.

---

### How the Components Interact
When a query hits `/api/search`:
1. **Extract:** `extract_query_constraints()` parses explicit structured constraints from the raw query (falling back to regex if the API fails).
2. **Filter:** `apply_hard_prefilter()` narrows the candidate pool to companies that strictly satisfy all explicit constraints.
3. **Retrieve:** `hybrid_search()` runs BM25 (exact lexical matching) and `all-MiniLM-L6-v2` (semantic similarity) over the surviving pool and fuses rankings via RRF to select the top-$k$ candidates.
4. **Rerank:** `rerank_candidates()` evaluates all top-$k$ candidates in one batched Gemini call, verifying constraints and industry/role fit with a 1–2 sentence reasoning step before assigning a final score.
5. **Serve:** Results above `min_score` are sorted descending and returned alongside observability metadata (`parsed_constraints`, `prefilter_active`, `prefilter_survivors`, `error_count`, `latency_ms`).

---

### Why I Chose This Design (Iterative Evolution)
* **Moving past Baseline A (Naive LLM):** I started with Baseline A using a local `llama3` model via Ollama. Evaluating all 477 companies sequentially was prohibitively slow and frequently timed out, forcing me to cap tests at 20–30 companies. Worse, on queries like *"Public software companies with over 1000 employees"*, unrelated companies received inflated scores (e.g., *European Drinks* scored `82`), while secondary IT providers were unfairly penalized.
* **Prompt Engineering:** Explicitly instructing the model to inspect `primary_naics`, `secondary_naics`, `core_offerings`, and `description`—and requiring a Chain-of-Thought reasoning step before scoring—dramatically improved accuracy, but could not solve the latency bottleneck of scanning all 477 records.
* **Why Hybrid Retrieval + RRF:** To narrow the pool to the top 10–50 candidates before calling the LLM, I combined BM25 and dense embeddings. Semantic search understands context but struggles with exact names, codes, and acronyms; lexical search catches exact terms but misses intent. Because BM25 scores (unbounded, typically `1–10+`) and cosine similarity (`0–1`) operate on different scales, RRF merges them cleanly by rank.
* **Why Hard Pre-Filtering:** Even with hybrid retrieval and soft score adjustments, companies with `< 1,000` employees still appeared when the user explicitly asked for `> 1,000`. Separating *explicit structured constraints* (hard filter) from *fuzzy industry intent* (hybrid retrieval + LLM reranking) eliminated these violations without hurting semantic discovery.
* **Why Gemini API + Batching:** Running PyTorch (`SentenceTransformer`) alongside a local Ollama model on an 8GB RAM machine with limited VRAM triggered `cudaMalloc` out-of-memory errors (`500 Internal Server Error`). Migrating the LLM layer to the Gemini API (`v1beta` JSON mode) eliminated local hardware bottlenecks, and batching all top-$k$ candidates into a single call avoided Free Tier rate limits (`429 Too Many Requests`) while cutting latency.

---

### Offline Evaluation & Metric Selection
To prove objectively that each stage improves upon the embedding-only baseline (**Baseline B**), I built `build_ground_truth.py` (which generates deterministic programmatic ground truth for 6 structured queries and flags subjective semantic queries as `TODO` for deliberate manual labeling) and `evaluate.py`.

**Why these metrics:**
* **Precision@k (`P@5`):** Directly reflects user-facing quality—what fraction of the top 5 results shown to the user are actually relevant.
* **Recall@k (`R@5`):** Measures coverage—what fraction of all true matches in the dataset were successfully surfaced in the top 5.
* **MRR (Mean Reciprocal Rank):** Measures how high the *first* relevant company appears (`1.000` means the #1 result is always correct), which is critical for search UX.

**Benchmark Results (averaged over 6 programmatic queries, `k = 5`):**

| System | P@5 | R@5 | MRR |
| :--- | :---: | :---: | :---: |
| **BM25 (lexical only)** | 0.433 | 0.070 | 0.450 |
| **Baseline B (embed only)** | 0.533 | 0.137 | 0.583 |
| **Hybrid RRF + Hard Filter** | 0.867 | 0.193 | 1.000 |
| **Hybrid + Gemini Rerank** | **0.933** | **0.203** | **1.000** |

* **Hybrid RRF vs. Baseline B:** `P@5` improved by **+63%** (`0.533 → 0.867`), `R@5` by **+41%** (`0.137 → 0.193`), and `MRR` reached **1.000** (first result correct on every query).
* **Impact of Gemini Reranking:** On *"Public software companies with more than 1000 employees"*, Baseline B scored `0.00 P@5` (returning small private HR firms), Hybrid RRF jumped to `0.80` (still letting `aeriepharma.com`—a large public pharma company—slip through), and Gemini Rerank removed the false positive to reach **`1.00 P@5`** (`epam.com`, `tcs.com`, `tietoevry.com`, `wipro.com`, `boozallen.com`). On *"Software and technology companies"*, Gemini similarly lifted `P@5` from `0.80` to **`1.00`**.

---

## 3.2 Tradeoffs

1. **Latency & Scalability over Exhaustive LLM Evaluation:**
   * *Decision:* Only the top-$k$ candidates from Hybrid RRF reach the LLM, rather than scoring all 477 companies.
   * *Tradeoff:* If a relevant company is missed by both BM25 and `all-MiniLM-L6-v2` (outside the top-50 pool), the reranker never sees it. Combining lexical and semantic retrieval maximizes first-stage recall to mitigate this.

2. **Constraint Precision over Recall on Missing Data:**
   * *Decision:* In the Hard Pre-Filter (and programmatic ground truth), companies with `null` values on an explicitly constrained field (e.g., `employee_count: null` when asking for `> 1000 employees`) are excluded.
   * *Tradeoff:* This may drop valid companies with incomplete records, but guarantees explicit user requirements are strictly respected rather than polluted by unverified guesses.

3. **API Rate-Limit Efficiency & Speed over Per-Company Isolation:**
   * *Decision:* Reranking all top-$k$ candidates in a single batched Gemini API call instead of one request per company.
   * *Tradeoff:* A single malformed response or API error affects the whole batch rather than one company, and very large `top_k` values could degrade LLM attention across candidates. Keeping `RERANK_TOP_K = 10` balances context quality, speed, and Free Tier quota limits.

4. **Startup Speed & Reliability over Disk Space:**
   * *Decision:* Caching `.npy` embeddings and `.pkl` BM25 indices on disk, invalidated via a SHA-256 hash of `companies.jsonl`.
   * *Tradeoff:* Uses a small amount of local disk space and requires bumping `INDEX_SCHEMA_VERSION` when serialization changes, in exchange for near-instant server restarts.

---

## 3.3 Error Analysis

### Where the System Struggles
1. **Supply-Chain Role Ambiguity (Supplier vs. End-Brand):** Both BM25 and `all-MiniLM-L6-v2` rely on surface/bi-encoder similarity. Queries like *"Companies supplying packaging for cosmetics brands"* share heavy vocabulary with cosmetics brands themselves; a packaging supplier using generic terms (*"molded plastics"*) may miss the top-50 retrieval pool.
2. **Strict NAICS Boundaries vs. Real-World Industry Fit:** Programmatic evaluation and industry filtering rely on `primary_naics`, which can be narrower than how a company actually operates.
3. **Incomplete Company Records:** Companies with `null` metadata on constrained fields are silently dropped by the hard pre-filter.
4. **Hard-Coded Location Dictionaries:** `COUNTRY_CODE` and `REGION_COUNTRIES` cover a fixed subset of countries (mostly Europe, US, China). Queries for unmapped countries fall back to raw substring matching, which fails if the address only stores an ISO code.

### Concrete Misclassification Examples
* **Baseline A (Naive LLM) — *European Drinks* (Score: 82 on *"Public software companies with over 1000 employees"*):** Without explicit attribute-checking rules and Chain-of-Thought reasoning, the local LLM latched onto company scale and ignored `primary_naics` / `core_offerings`.
* **Stage 1 Retrieval Noise — `aeriepharma.com`:** On *"Public software companies with more than 1000 employees"*, `aeriepharma.com` passed the hard pre-filter (public, >1000 employees) and entered the top-5 Hybrid RRF pool due to keyword overlap, before the Stage 2 Gemini reranker correctly eliminated it.
* **Taxonomy Mismatch — `paychex.de` & `iml4e.org` (on *"Software companies in Germany"*, `0.60 P@5`):** Both survived the Germany filter and were ranked in the top 5 by Gemini (`paychex.de` provides payroll/HR software; `iml4e.org` is an AI/ML research initiative), but neither carried one of the 4 strict `SOFTWARE_NAICS` labels in the ground truth.
* **Cross-Domain Keyword Overlap — *Compania Națională de Căi Ferate CFR* & *Central de Envasados*:** Large industrial/logistics companies occasionally enter the top-10 candidate pool due to overlapping corporate/operational terms, occupying slots in `RERANK_TOP_K = 10` even though Gemini demotes them to `0–20`.

---

## 3.4 Scaling (500 → 100,000+ Companies)

At 477 companies, brute-force cosine similarity (`util.cos_sim` over NumPy) and in-memory `BM25Okapi` take milliseconds. At **100,000+ companies** (or millions), I would make five architectural changes:

1. **Move to a Dedicated Hybrid Search Engine (e.g., Qdrant, OpenSearch/Elasticsearch):** Replace $O(N)$ in-memory scans with an **HNSW (Approximate Nearest Neighbor)** vector index, an inverted BM25 index, and native B-tree/keyword indexes on structured metadata (`employee_count`, `revenue`, `is_public`, `country_code`).
2. **True Pre-Filtered Retrieval:** Today, `hybrid_search()` scores all companies, slices `RETRIEVAL_POOL_N = 50`, and *then* intersects with `allowed` indices. At 100,000 companies, a rare filter (e.g., *"public companies in Romania"*) could have zero overlap with the global top-50. Filtering must be pushed *inside* the index (native filtered HNSW / `bool.filter`).
3. **Add a Fast Cross-Encoder Stage:** Use a 3-tier funnel:
   * **Stage 1 (Filtered Hybrid DB):** 100,000 → top 200 (~20ms).
   * **Stage 2 (Cross-Encoder, e.g., `ms-marco-MiniLM-L-6-v2`):** 200 → top 15–20 (~100ms on GPU).
   * **Stage 3 (LLM Reranker):** Only top 15–20 go to Gemini for final qualification and human-readable reasoning.
4. **Incremental Indexing & Query Caching:** Replace full-file SHA-256 rebuilds with incremental record upserts and add a Redis cache for frequent queries and extracted constraints.
5. **Async Serving:** Replace Flask's dev server with FastAPI + Uvicorn (async I/O) and decouple indexing workers from live search traffic.

---

## 3.5 Failure Modes & Production Monitoring

### When the System Produces Confident but Incorrect Results
1. **Silent Constraint Extraction Failures:** If the LLM misparses a query (e.g., misses a negation like *"non-US companies"* or misinterprets a range like *"founded in the 90s"*), the hard pre-filter executes cleanly with high confidence on the wrong slice of data.
2. **Stale or Marketing-Heavy Descriptions:** Because retrieval and reranking rely heavily on `primary_naics`, `core_offerings`, and `description`, a traditional company with buzzword-heavy marketing (*"AI-powered"*) can receive a high score, while a pivoted SaaS company with an outdated NAICS code can be confidently rejected.
3. **Batch Attention Drift ("Lost in the Middle"):** Sending too many profiles in a single batched LLM prompt can cause the model to produce plausible reasoning while subtly mixing up attributes across indices.
4. **Regex Fallback Blind Spots:** If Gemini fails during constraint extraction and falls back to `regex_fallback_constraints()`, nuanced phrasing (*"at least 1k staff"*, *"DACH region"*) is silently ignored, turning a strict query into an unfiltered search.

### What to Monitor in Production
* **Pipeline Funnel Metrics:** Track `prefilter_survivors`, `candidates_evaluated`, and **Zero-Result Rate** per query. Spikes in zero-result queries signal overly aggressive constraint extraction or broken metadata parsing.
* **Fallback & Error Rates:** Alert on spikes in `regex_fallback` triggers, `error_count > 0`, `429` rate-limit responses, or `missing_in_batch` items.
* **Score Distribution Drift:** Monitor the distribution of LLM scores (`0–100`) over time to catch prompt regressions or model version drift.
* **Latency Breakdown (p50 / p95 / p99):** Log latency separately for constraint extraction, retrieval, and LLM reranking—not just total `latency_ms`.
* **Human-in-the-Loop Audits:** Weekly sampling of high-confidence (`>= 80`) and borderline (`40–60`) results, feeding misclassifications back into `eval_set.jsonl`.

---

## Observations & Next Priorities

* **Where the system works extremely well:** Queries combining explicit structured filters (size, public/private, known country) with clear industry intent—achieving `0.933 P@5` and `1.000 MRR` on the programmatic benchmark.
* **Key assumptions made:** Company metadata (`employee_count`, `is_public`, `address`, `primary_naics`) is reasonably accurate; `null` on a constrained field should fail closed; and `all-MiniLM-L6-v2` + BM25 surfaces relevant candidates within the top-50 pool.
* **Top next improvements:**
  1. **True pre-filtered retrieval** (filter before `[:pool_n]` slicing) so rare constraints never lose recall.
  2. **Expand `eval_set.jsonl`** by hand-labeling the `TODO` semantic/supply-chain queries and adding negation/edge-case tests.
  3. **Standardize geographic resolution** (replace the manual `COUNTRY_CODE` dict with `pycountry` + region mapping).
  4. **Introduce a lightweight Cross-Encoder** before the LLM reranker to improve precision on nuanced B2B relationships without increasing API costs.