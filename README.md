# BSI RAG Embedding Pipeline

Case report generation, embedding, FAISS index construction, and nearest-neighbour retrieval scripts for the RAG-LLM arm described in:

> **Dual-Timepoint Machine Learning and Retrieval-Augmented Generation for 30-Day Mortality Prediction in Bloodstream Infection: A Multi-Cohort Validation Study**

---

## Overview

This pipeline converts structured clinical data into de-identified natural-language case reports, embeds them using the OpenAI API, builds a FAISS index for similarity search, and retrieves nearest neighbours for RAG-based LLM inference.

```
admission_level_base.csv
bsi_with_ast_tt.csv
        │
        ▼
make_case_reports.py       → reports/*.report.md  (de-identified, outcome-stripped)
        │
        ▼
sanitize_reports.py        → reports_sanitized/*.report.md  (outcome section removed)
        │
        ▼
embed_case_reports.py      → embeddings_out/embeddings_shards/*.npy
                                             case_report_embeddings_metadata.csv
        │
        ▼
build_corpus_embeddings_and_faiss.py  → rag_index_openai/{early,ast}/
                                          *_faiss.index
                                          *_embeddings.npy
                                          *_corpus_metadata.csv
        │
        ▼
retrieve_neighbors.py      → rag_retrieval_openai/{early,ast}_{internal,external}/
                               retrieval_summary.csv
                               retrieval_long.csv
```

---

## Requirements

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

Python 3.9+ recommended.

---

## Scripts

### 1. `make_case_reports.py`
Generates de-identified, outcome-agnostic clinical case reports in Markdown format.

```bash
# Single admission
python make_case_reports.py \
  --data-dir ./data \
  --admission-id 12345 \
  --out-dir ./reports_single

# All admissions
python make_case_reports.py \
  --data-dir ./data \
  --all \
  --out-dir ./reports_all
```

**Input files in `--data-dir`:**
- `admission_level_base.csv` — admission-level feature table (demographics, labs, severity scores, etc.)
- `bsi_with_ast_tt.csv` — episode-level microbiology and AST results

### 2. `sanitize_reports.py`
Strips the `[Outcome]` section and any outcome-leaking keywords from case reports to prevent label leakage before embedding.

```bash
python sanitize_reports.py \
  --in_dir ./reports_all \
  --out_dir ./reports_sanitized
```

### 3. `embed_case_reports.py`
Embeds `.report.md` files using the OpenAI embeddings API. Supports sharded saving and resumable runs.

```bash
python embed_case_reports.py \
  --report-dir ./reports_sanitized \
  --out-dir ./embeddings_out \
  --model text-embedding-3-large \
  --batch-size 64 \
  --shard-size 2048 \
  --resume \
  --qc
```

### 4. `build_corpus_embeddings_and_faiss.py`
Builds OpenAI embeddings and FAISS `IndexFlatIP` indexes for the development corpus (Early and AST timepoints). Vectors are L2-normalized for cosine similarity via inner product.

```bash
python build_corpus_embeddings_and_faiss.py \
  --early-corpus ./llm_templates_cleaned/rag_corpus_early_development.jsonl \
  --ast-corpus ./llm_templates_cleaned/rag_corpus_ast_development.jsonl \
  --output-dir ./rag_index_openai \
  --embedding-model text-embedding-3-large
```

### 5. `retrieve_neighbors.py`
Embeds query episodes and retrieves top-k nearest neighbours from the corpus FAISS index. Computes unweighted and similarity-weighted neighbourhood mortality risk.

```bash
python retrieve_neighbors.py \
  --early-index-dir ./rag_index_openai/early \
  --ast-index-dir ./rag_index_openai/ast \
  --early-internal-query ./llm_templates_cleaned/rag_queries_early_internal_temporal_test.jsonl \
  --early-external-query ./llm_templates_cleaned/rag_queries_early_external.jsonl \
  --ast-internal-query ./llm_templates_cleaned/rag_queries_ast_internal_temporal_test.jsonl \
  --ast-external-query ./llm_templates_cleaned/rag_queries_ast_external.jsonl \
  --eval-labels-csv ./llm_templates_cleaned/template_eval_labels.csv \
  --output-dir ./rag_retrieval_openai \
  --embedding-model text-embedding-3-large \
  --top-k 5
```

---

## Data Availability

Patient-level data cannot be shared due to institutional data governance requirements. The pipeline is designed to operate on any dataset conforming to the input schema described above. Column names and expected formats are documented in the script docstrings.

---

## Citation

If you use this code, please cite:

> [citation to be added upon publication]

---

## License

MIT License
