# Narrative Embeddings for Bloodstream Infection Phenotyping

This repository contains the analysis code for "Evaluating Narrative Embeddings of Routine Hospital Data for Bloodstream Infection Phenotyping: Retrospective Cohort Study." The pipeline converts structured hospitalization-level bloodstream infection data into standardized outcome-stripped case narratives, generates narrative embeddings, applies UMAP-HDBSCAN clustering, performs structured-variable comparator analyses, supports phenotype characterization and interpretability analyses, conducts within-pathogen sensitivity analyses, and produces in-hospital mortality trajectory analyses.

---

## Analysis Overview

1. **Narrative construction** (`make_case_narratives.py`): Convert structured admission-level BSI data into de-identified, outcome-stripped natural-language case narratives.
2. **Narrative embedding** (`embed_narratives.py`): Embed narratives using the OpenAI embeddings API.
3. **UMAP + HDBSCAN clustering** (`cluster_embeddings_umap_hdbscan.py`): Apply UMAP dimensionality reduction and HDBSCAN clustering to identify BSI phenotype clusters.
4. **Structured-variable comparator clustering** (`structured_variable_clustering_comparators.py`): Run PCA + k-means and structured UMAP + HDBSCAN on structured variables; assess concordance with embedding-derived phenotypes.
5. **Phenotype interpretability** (`phenotype_interpretability.py`): Characterize phenotype clusters using SHAP values and permutation feature importance.
6. **Supervised back-mapping** (`supervised_backmapping.py`): Interpretability audit — train classifiers to recapitulate embedding-derived phenotype labels from structured variables.
7. **Within-pathogen sensitivity analysis** (`ecoli_within_pathogen_analysis.py`): Repeat UMAP + HDBSCAN phenotyping restricted to E. coli BSI admissions.
8. **Mortality trajectory analysis** (`mortality_trajectory_analysis.py`): Descriptive in-hospital mortality trajectory analysis by phenotype cluster, with discharge treated as a competing event.
9. **Figures and tables** (`make_figures_tables.py`): Generate all manuscript figures and tables.
10. **Consistency check** (`check_repository_consistency.py`): Verify repository alignment with phenotyping manuscript.

---

## UMAP Clarification

UMAP is used in two separate roles in this pipeline:

- **3D UMAP for HDBSCAN clustering**: HDBSCAN cluster assignment is performed on a 3-dimensional UMAP representation of the narrative embedding space (controlled by `--umap-n-components-cluster`, default 3).
- **2D UMAP for visualization only**: A separate 2-dimensional UMAP projection is generated solely for visual display (controlled by `--umap-n-components-viz`, default 2).
- **The displayed 2-dimensional UMAP projection is NOT used for cluster assignment.**

This applies to both the main embedding-derived phenotyping pipeline (`cluster_embeddings_umap_hdbscan.py`) and the structured-variable comparator clustering (`structured_variable_clustering_comparators.py`).

---

## Identifier and Outcome Exclusion Clarifications

- **Pseudonymized identifiers** (`newpatient_ID`) are used only for output file naming and metadata linkage. They are not included in the narrative text submitted to the embedding model.
- **The following variables are excluded from narrative text** to prevent label leakage before embedding: mortality (30-day, in-hospital), death date, discharge status, length of stay, follow-up duration, time-to-event variables, embedding-derived variables, and cluster labels.
- Only pre-outcome clinical data available at the time of the BSI index culture are included in the narrative.

---

## Repository Structure

| Script | Description |
|--------|-------------|
| `make_case_narratives.py` | Convert structured BSI data to de-identified, outcome-stripped narratives |
| `embed_narratives.py` | Embed narratives via OpenAI embeddings API |
| `cluster_embeddings_umap_hdbscan.py` | UMAP (3D for clustering, 2D for visualization) + HDBSCAN phenotyping |
| `structured_variable_clustering_comparators.py` | Structured-variable comparator clustering (PCA+k-means, UMAP+HDBSCAN); concordance with embedding phenotypes |
| `phenotype_interpretability.py` | SHAP and permutation feature importance for phenotype characterization |
| `supervised_backmapping.py` | Interpretability audit: classify embedding-derived phenotypes from structured variables |
| `ecoli_within_pathogen_analysis.py` | Within-pathogen sensitivity analysis restricted to E. coli BSI |
| `mortality_trajectory_analysis.py` | Descriptive in-hospital mortality trajectories by phenotype; Gray test; cause-specific Cox |
| `make_figures_tables.py` | Generate all manuscript figures and tables |
| `check_repository_consistency.py` | Heuristic consistency checker for repository-manuscript alignment |
| `requirements.txt` | Python package dependencies |
| `legacy_rag_prediction/` | Legacy RAG-LLM mortality-prediction scripts (see below) |

---

## Pipeline Diagram

```
admission_level_base.csv
bsi_with_ast_tt.csv
        |
        v
make_case_narratives.py        --> narratives/*.narrative.md
  [de-identified, outcome-stripped; pseudo-ID in filename only]
        |
        v
embed_narratives.py            --> embeddings_out/embeddings_shards/*.npy
                                   narrative_embeddings_metadata.csv
        |
        v
cluster_embeddings_umap_hdbscan.py
  [3D UMAP --> HDBSCAN (cluster assignment)]
  [2D UMAP --> visualization only]
        |                      --> clustering_out/
        |                          umap_3d_for_clustering.csv
        |                          umap_2d_for_visualization.csv
        |                          hdbscan_labels.csv
        |                          cluster_summary.csv
        |
        |----> structured_variable_clustering_comparators.py
        |        [PCA+k-means; structured UMAP+HDBSCAN; ARI/NMI concordance]
        |      --> comparators_out/
        |
        |----> phenotype_interpretability.py
        |      --> interpretability_out/
        |
        |----> supervised_backmapping.py
        |        [interpretability audit: classify phenotypes from structured vars]
        |      --> backmapping_out/
        |
        |----> ecoli_within_pathogen_analysis.py
        |        [within-pathogen sensitivity analysis]
        |      --> ecoli_out/
        |
        |----> mortality_trajectory_analysis.py
        |        [descriptive: cumulative incidence, Gray test, cause-specific Cox]
        |      --> trajectories_out/
        |
        v
make_figures_tables.py         --> figures_tables/
```

---

## Scripts

### 1. `make_case_narratives.py`

Converts structured BSI admission-level data into de-identified, outcome-stripped natural-language case narratives. Outcome fields and patient pseudo-identifiers are excluded from the narrative text; pseudo-IDs are used only for file naming.

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

Input files in `--data-dir`: `admission_level_base.csv`, `bsi_with_ast_tt.csv`

---

### 2. `embed_narratives.py`

Embeds `.narrative.md` files using the OpenAI embeddings API. Supports sharded saving and resumable runs.

```bash
python embed_narratives.py \
  --narrative-dir ./narratives \
  --out-dir ./embeddings_out \
  --model text-embedding-3-large \
  --batch-size 64 \
  --shard-size 2048 \
  --resume
```

---

### 3. `cluster_embeddings_umap_hdbscan.py`

UMAP dimensionality reduction and HDBSCAN clustering. HDBSCAN operates on 3D UMAP coordinates; a separate 2D UMAP is generated for visualization only.

```bash
python cluster_embeddings_umap_hdbscan.py \
  --embeddings-dir ./embeddings_out \
  --metadata-csv ./embeddings_out/narrative_embeddings_metadata.csv \
  --out-dir ./clustering_out \
  --outcome-csv ./data/admission_level_base.csv \
  --umap-n-components-cluster 3 \
  --umap-n-components-viz 2 \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1 \
  --hdbscan-min-cluster-size 50 \
  --hdbscan-min-samples 10 \
  --seed 42
```

---

### 4. `structured_variable_clustering_comparators.py`

Runs PCA + k-means and structured-variable UMAP + HDBSCAN comparator clustering, then assesses concordance with embedding-derived phenotype labels using ARI and NMI. Descriptive comparator analyses only.

```bash
python structured_variable_clustering_comparators.py \
  --feature-csv ./data/admission_level_base.csv \
  --cluster-labels ./clustering_out/hdbscan_labels.csv \
  --out-dir ./comparators_out \
  --n-pca-components 50 \
  --kmeans-k-range 2,3,4,5,6,7,8 \
  --seed 42
```

---

### 5. `phenotype_interpretability.py`

Characterizes phenotype clusters using SHAP values and permutation feature importance.

```bash
python phenotype_interpretability.py \
  --feature-csv ./data/admission_level_base.csv \
  --cluster-labels ./clustering_out/hdbscan_labels.csv \
  --embeddings-dir ./embeddings_out \
  --out-dir ./interpretability_out \
  --n-background 500
```

---

### 6. `supervised_backmapping.py`

Interpretability audit: trains LightGBM with stratified CV to recapitulate embedding-derived phenotype labels from structured clinical variables. Not a prognostic prediction model.

```bash
python supervised_backmapping.py \
  --feature-csv ./data/admission_level_base.csv \
  --cluster-labels ./clustering_out/hdbscan_labels.csv \
  --out-dir ./backmapping_out \
  --cv-folds 5 \
  --seed 42
```

---

### 7. `ecoli_within_pathogen_analysis.py`

Within-pathogen sensitivity analysis restricted to E. coli BSI admissions. Applies the same UMAP + HDBSCAN framework to pre-generated embeddings for the E. coli subset.

```bash
python ecoli_within_pathogen_analysis.py \
  --metadata-csv ./embeddings_out/narrative_embeddings_metadata.csv \
  --embeddings-dir ./embeddings_out \
  --feature-csv ./data/admission_level_base.csv \
  --out-dir ./ecoli_out \
  --ecoli-organism-col BSI_episode_short_clean \
  --ecoli-string "Escherichia coli"
```

---

### 8. `mortality_trajectory_analysis.py`

Descriptive in-hospital mortality trajectory analysis by phenotype cluster, with discharge alive treated as a competing event. Outputs cumulative incidence, Gray test, and cause-specific Cox results.

Column names must be adapted to your local deidentified dataset.

```bash
python mortality_trajectory_analysis.py \
  --feature-csv ./data/admission_level_base.csv \
  --cluster-labels ./clustering_out/hdbscan_labels.csv \
  --out-dir ./trajectories_out \
  --time-col days_from_index_bsi_to_death_or_discharge \
  --event-col inhospital_mortality \
  --competing-col discharge_alive \
  --landmark-day 30 \
  --reference-cluster 0
```

---

### 9. `make_figures_tables.py`

Generates all manuscript figures and tables.

```bash
python make_figures_tables.py \
  --clustering-dir ./clustering_out \
  --interpretability-dir ./interpretability_out \
  --trajectories-dir ./trajectories_out \
  --comparators-dir ./comparators_out \
  --feature-csv ./data/admission_level_base.csv \
  --supervised-backmapping-dir ./backmapping_out \
  --ecoli-dir ./ecoli_out \
  --out-dir ./figures_tables \
  --dpi 300 \
  --format pdf
```

Note: Figure 1 (study workflow schematic) requires manual assembly and is not generated by this script.

---

### 10. `check_repository_consistency.py`

Heuristic consistency checker.

```bash
python check_repository_consistency.py --repo-dir .
```

---

## Legacy Scripts

Legacy scripts from a previous RAG-LLM mortality-prediction project are retained in `legacy_rag_prediction/` for transparency and were not used for the present phenotyping analyses.

---

## Reproducibility and Data Availability

- Patient-level clinical data are not included because of institutional data governance and privacy restrictions.
- Scripts are designed to run on a deidentified hospitalization-level analytic dataset with manuscript-specified columns.
- Outcome-related variables and identifiers are excluded from embedding input.
- Clustering parameters are specified before outcome characterization and should not be optimized to maximize mortality separation.

---

## Citation

If you use this code, please cite:

> [citation to be added upon publication]

Analysis code archived on Zenodo: DOI [10.5281/zenodo.19702160](https://doi.org/10.5281/zenodo.19702160)

Code repository: https://github.com/tcmhwd/bsi-rag-embedding-pipeline

---

## License

MIT License
