# Release Notes

## v2.0.0 — Phenotyping manuscript release

This release aligns the repository with the phenotyping manuscript:
"Evaluating Narrative Embeddings of Routine Hospital Data for Bloodstream Infection Phenotyping: Retrospective Cohort Study."

### Changes from v1.x
- Repository scope changed from RAG-LLM mortality prediction to narrative embedding phenotyping.
- Legacy RAG/FAISS/nearest-neighbour scripts moved to `legacy_rag_prediction/`.
- New analysis scripts added covering the full phenotyping pipeline.
- README updated to reflect phenotyping manuscript scope.
- Pseudo-identifiers and outcome variables excluded from embedding narrative text.
- UMAP roles clarified: 3D UMAP for HDBSCAN clustering, separate 2D UMAP for visualization only.

### Before creating this release
1. Confirm README reflects the phenotyping manuscript.
2. Confirm pseudo-identifiers and outcome variables are not included in embedding narratives.
3. Confirm HDBSCAN clustering uses 3D UMAP coordinates and 2D UMAP is visualization only.
4. Run `python check_repository_consistency.py` and resolve any failures.
5. Create a GitHub release named: `v2.0.0 — Phenotyping manuscript release`
6. Archive the release on Zenodo.
7. Update the manuscript Code Availability statement with the version-specific Zenodo DOI.

## v1.x — RAG-LLM mortality-prediction pipeline (legacy)

Initial release. Scripts for RAG-LLM dual-timepoint 30-day BSI mortality prediction.
See `legacy_rag_prediction/` for retained scripts.
