# Narrative Embeddings for Bloodstream Infection Phenotyping

This repository contains the analysis code for "Evaluating Narrative Embeddings of Routine Hospital Data for Bloodstream Infection Phenotyping: Retrospective Cohort Study." The pipeline converts structured hospitalization-level bloodstream infection data into standardized outcome-stripped case narratives, generates narrative embeddings, applies UMAP-HDBSCAN clustering, performs structured-variable comparator analyses, supports phenotype characterization and interpretability analyses, and produces mortality trajectory analyses.

---

## Overview

```
admission_level_base.csv
bsi_with_ast_tt.csv
        │
        ▼
make_case_narratives.py        → narratives/*.narrative.md  (de-identified, outcome-stripped)
        │
        ▼
embed_narratives.py            → embeddings_out/embeddings_shards/*.npy
                                              narrative_embeddings_metadata.csv
        │
        ▼
cluster_embeddings_umap_hdbscan.py  → clustering_out/
                                         umap_2d.npy / umap_3d.npy
                                         hdbscan_labels.csv
        │
        ▼
structured_variable_comparators.py  → comparators_out/
                                         comparator_cv_results.csv
        │
        ├──▶ phenotype_interpretability.py  → interpretability_out/
        │                                       shap_values.npy
        │                                       pfi_results.csv
        │
        └──▶ mortality_trajectory_analysis.py → trajectories_out/
                                                  competing_risk_results.csv
        │
        ▼
make_figures_tables.py         → figures/ + tables/
```

---

## Requirements

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

Python 3.10+ recommended.

---

## Scripts

### 1. `make_case_narratives.py`
Converts structured BSI admission-level data into de-identified, outcome-stripped natural-language case narratives. Uses a standardized template covering demographics, comorbidities, severity scores, microbiology, and antibiotic susceptibility. Outcome fields are excluded to prevent label leakage.

```bash
# Single admission
python make_case_narratives.py \
  --data-dir ./data \
  --admission-id 12345 \
  --out-dir ./narratives

# All admissions
python make_case_narratives.py \
  --data-dir ./data \
  --all \
  --out-dir ./narratives
```

**Input files in `--data-dir`:**
- `admission_level_base.csv` — admission-level feature table (demographics, labs, severity scores, comorbidities, organ support)
- `bsi_with_ast_tt.csv` — episode-level microbiology and antibiotic susceptibility testing results

### 2. `embed_narratives.py`
Embeds `.narrative.md` files using the OpenAI embeddings API. Supports sharded saving and resumable runs. Produces sharded `.npy` embedding arrays and a metadata CSV linking each embedding to its source admission.

```bash
python embed_narratives.py \
  --narrative-dir ./narratives \
  --out-dir ./embeddings_out \
  --model text-embedding-3-large \
  --batch-size 64 \
  --shard-size 2048 \
  --resume \
  --qc
```

### 3. `cluster_embeddings_umap_hdbscan.py`
Loads narrative embeddings, applies UMAP for dimensionality reduction (2D and 3D projections), then applies HDBSCAN clustering to identify BSI phenotype clusters. Outputs UMAP coordinates and cluster labels. Supports grid search over HDBSCAN hyperparameters.

```bash
python cluster_embeddings_umap_hdbscan.py \
  --embeddings-dir ./embeddings_out \
  --metadata-csv ./embeddings_out/narrative_embeddings_metadata.csv \
  --outcome-csv ./data/admission_level_base.csv \
  --out-dir ./clustering_out \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1 \
  --hdbscan-min-cluster-size 50 \
  --hdbscan-min-samples 10
```

### 4. `structured_variable_comparators.py`
Trains and cross-validates structured-variable comparator models (logistic regression, gradient-boosted trees) on the same admission-level feature set used for narrative construction. Compares predictive performance against embedding-based approaches using identical cross-validation splits. Outputs cross-validation results and feature importance.

```bash
python structured_variable_comparators.py \
  --feature-csv ./data/admission_level_base.csv \
  --cluster-labels ./clustering_out/hdbscan_labels.csv \
  --out-dir ./comparators_out \
  --cv-folds 5 \
  --seed 42
```

### 5. `phenotype_interpretability.py`
Characterizes identified phenotype clusters using SHAP values and permutation feature importance (PFI). Produces per-cluster feature attribution summaries and beeswarm/waterfall plots. Supports both embedding-space interpretation and structured-variable interpretation of clusters.

```bash
python phenotype_interpretability.py \
  --feature-csv ./data/admission_level_base.csv \
  --cluster-labels ./clustering_out/hdbscan_labels.csv \
  --embeddings-dir ./embeddings_out \
  --out-dir ./interpretability_out \
  --n-background 500
```

### 6. `mortality_trajectory_analysis.py`
Performs competing-risk survival analysis of 30-day mortality trajectories across identified phenotype clusters. Uses the Fine-Gray subdistribution hazard model to account for competing events. Outputs cumulative incidence curves, subdistribution hazard ratios, and cluster-level survival summaries.

```bash
python mortality_trajectory_analysis.py \
  --feature-csv ./data/admission_level_base.csv \
  --cluster-labels ./clustering_out/hdbscan_labels.csv \
  --out-dir ./trajectories_out \
  --time-col days_to_death_or_discharge \
  --event-col mortality_30d \
  --competing-col discharge_alive_30d
```

### 7. `make_figures_tables.py`
Generates all main-text and supplementary figures and tables from intermediate outputs. Produces UMAP scatter plots colored by cluster and clinical variables, Kaplan-Meier / cumulative incidence plots, SHAP summary plots, structured-variable comparator performance tables, and cluster characterization heatmaps.

```bash
python make_figures_tables.py \
  --clustering-dir ./clustering_out \
  --interpretability-dir ./interpretability_out \
  --trajectories-dir ./trajectories_out \
  --comparators-dir ./comparators_out \
  --out-dir ./figures_tables
```

---

## Legacy RAG-LLM Scripts

The following scripts from the original RAG-LLM pipeline are retained for reference:

| Script | Description |
|---|---|
| `make_case_reports.py` | Generates de-identified case reports (RAG corpus) |
| `sanitize_reports.py` | Strips outcome sections before embedding |
| `embed_case_reports.py` | Embeds reports for FAISS index construction |
| `build_corpus_embeddings_and_faiss.py` | Builds FAISS index for nearest-neighbour retrieval |
| `retrieve_neighbors.py` | Retrieves top-k similar cases for RAG-LLM inference |

---

## Data Availability

Patient-level data cannot be shared due to institutional data governance requirements. The pipeline is designed to operate on any dataset conforming to the input schema described above. Column names and expected formats are documented in the script docstrings.

---

## Citation

If you use this code, please cite:

> [citation to be added upon publication]

Analysis code archived on Zenodo: DOI [10.5281/zenodo.19702160](https://doi.org/10.5281/zenodo.19702160)

Code repository: https://github.com/tcmhwd/bsi-rag-embedding-pipeline

---

## License

MIT License
