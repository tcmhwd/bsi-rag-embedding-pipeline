#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ecoli_within_pathogen_analysis.py

Within-pathogen sensitivity analysis restricted to Escherichia coli bloodstream
infection (BSI) hospitalisations. Evaluates whether narrative embedding-based
phenotyping can reflect clinically meaningful heterogeneity beyond organism
identity alone. This analysis is exploratory.

Pipeline:
  1. Subset admissions to those with E. coli BSI (index or primary episode).
  2. Reuse pre-generated narrative embeddings (no re-embedding required if
     embeddings already cover these admissions).
  3. Apply the same UMAP + HDBSCAN framework:
       - 3D UMAP for HDBSCAN clustering (cluster assignment).
       - Separate 2D UMAP for visualization only.
  4. Output E. coli sub-phenotype labels and visualization coordinates.
  5. Perform descriptive characterization of E. coli sub-phenotypes.

Outputs (in --out-dir):
  - ecoli_subset_metadata.csv
  - ecoli_umap_3d_for_clustering.csv  (admission_ID, umap_c1, umap_c2, umap_c3)
  - ecoli_umap_2d_for_visualization.csv (admission_ID, umap_x, umap_y)
  - ecoli_hdbscan_labels.csv          (admission_ID, cluster_label, cluster_probability)
  - ecoli_cluster_summary.csv

Usage:
  python ecoli_within_pathogen_analysis.py \\
    --metadata-csv ./embeddings_out/narrative_embeddings_metadata.csv \\
    --embeddings-dir ./embeddings_out \\
    --feature-csv ./data/admission_level_base.csv \\
    --out-dir ./ecoli_out \\
    --ecoli-organism-col BSI_episode_short_clean \\
    --ecoli-string "Escherichia coli" \\
    --umap-n-neighbors 15 \\
    --umap-min-dist 0.1 \\
    --hdbscan-min-cluster-size 20 \\
    --hdbscan-min-samples 5 \\
    --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Within-pathogen sensitivity analysis: UMAP+HDBSCAN phenotyping "
            "restricted to E. coli BSI admissions. This is an exploratory analysis."
        )
    )
    p.add_argument("--metadata-csv", type=str, required=True,
                   help="narrative_embeddings_metadata.csv with admission_ID column")
    p.add_argument("--embeddings-dir", type=str, required=True,
                   help="Directory containing embeddings_shards/ subdirectory")
    p.add_argument("--feature-csv", type=str, required=True,
                   help="admission_level_base.csv for E. coli identification and characterization")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write E. coli analysis outputs")
    p.add_argument(
        "--ecoli-organism-col", type=str,
        default="BSI_episode_short_clean",
        help="Column used to identify E. coli episodes (default: BSI_episode_short_clean)",
    )
    p.add_argument(
        "--ecoli-string", type=str,
        default="Escherichia coli",
        help="String to match for E. coli subset (default: 'Escherichia coli')",
    )
    p.add_argument("--umap-n-neighbors", type=int, default=15,
                   help="UMAP n_neighbors (default: 15)")
    p.add_argument("--umap-min-dist", type=float, default=0.1,
                   help="UMAP min_dist (default: 0.1)")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=20,
                   help="HDBSCAN min_cluster_size (default: 20)")
    p.add_argument("--hdbscan-min-samples", type=int, default=5,
                   help="HDBSCAN min_samples (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for UMAP reproducibility (default: 42)")
    return p.parse_args()


def identify_ecoli_admissions(feature_csv: Path, ecoli_organism_col: str,
                               ecoli_string: str) -> set:
    """Identify admission_IDs with E. coli BSI from the feature CSV.

    Returns a set of admission_IDs where the organism column contains
    the specified E. coli string (case-insensitive substring match).
    """
    feat_df = pd.read_csv(feature_csv, low_memory=False)
    feat_df["admission_ID"] = pd.to_numeric(feat_df["admission_ID"], errors="coerce")

    if ecoli_organism_col not in feat_df.columns:
        print(
            f"[ERROR] Column '{ecoli_organism_col}' not found in feature CSV. "
            f"Use --ecoli-organism-col to specify the correct column name.",
            file=sys.stderr,
        )
        sys.exit(1)

    mask = feat_df[ecoli_organism_col].astype(str).str.contains(
        ecoli_string, case=False, na=False
    )
    ecoli_ids = set(feat_df.loc[mask, "admission_ID"].dropna().astype(int).tolist())
    print(f"[INFO] E. coli admissions identified: n={len(ecoli_ids)} "
          f"(matched '{ecoli_string}' in '{ecoli_organism_col}')")
    return ecoli_ids


def load_ecoli_embeddings(metadata_csv: Path, embeddings_dir: Path,
                           ecoli_admission_ids: set):
    """Load and subset narrative embeddings to E. coli admissions.

    Parameters
    ----------
    metadata_csv : Path
        narrative_embeddings_metadata.csv with admission_ID column.
    embeddings_dir : Path
        Directory containing embeddings_shards/ subdirectory.
    ecoli_admission_ids : set
        Set of admission_IDs to retain.

    Returns
    -------
    X_ecoli : np.ndarray
        Embedding matrix for E. coli admissions only.
    meta_ecoli_df : pd.DataFrame
        Metadata rows for E. coli admissions, aligned with X_ecoli.
    """
    meta_df = pd.read_csv(metadata_csv)
    meta_df["admission_ID"] = pd.to_numeric(meta_df["admission_ID"], errors="coerce")

    shard_dir = embeddings_dir / "embeddings_shards"
    shard_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not shard_paths:
        print(f"[ERROR] No shard files found in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    arrays = [np.load(sp) for sp in shard_paths]
    X_all = np.concatenate(arrays, axis=0).astype(np.float32)
    print(f"[INFO] Loaded all embeddings: shape={X_all.shape} from {len(shard_paths)} shard(s)")

    if len(meta_df) != X_all.shape[0]:
        print(
            f"[WARN] Metadata rows ({len(meta_df)}) != embedding rows ({X_all.shape[0]}). "
            "Ensure metadata and embedding shards are aligned.",
            file=sys.stderr,
        )

    # Subset to E. coli admissions
    ecoli_mask = meta_df["admission_ID"].isin(ecoli_admission_ids)
    meta_ecoli_df = meta_df[ecoli_mask].reset_index(drop=True)
    X_ecoli = X_all[ecoli_mask.values]

    print(f"[INFO] E. coli embedding subset: shape={X_ecoli.shape}, "
          f"n_admissions={len(meta_ecoli_df)}")

    if len(meta_ecoli_df) == 0:
        print("[ERROR] No E. coli admissions found in embedding metadata. "
              "Check that admission IDs match between feature CSV and metadata CSV.",
              file=sys.stderr)
        sys.exit(1)

    return X_ecoli, meta_ecoli_df


def run_umap(X: np.ndarray, n_components: int, n_neighbors: int, min_dist: float,
             seed: int, label: str) -> np.ndarray:
    """Run a single UMAP model and return the embedding array.

    Parameters
    ----------
    label : str
        Descriptive label printed in log output.
    """
    try:
        import umap as umap_lib
    except ImportError:
        print("[ERROR] umap-learn not installed. Run: pip install umap-learn", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Running UMAP [{label}]: n_components={n_components}, "
          f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric=cosine, seed={seed}")
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=True,
    )
    embedding = reducer.fit_transform(X)
    print(f"[INFO] UMAP [{label}] complete: output shape={embedding.shape}")
    return embedding


def run_hdbscan(X_3d: np.ndarray, min_cluster_size: int, min_samples: int):
    """Run HDBSCAN on 3D UMAP coordinates.

    Returns (labels, probabilities).
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        print("[ERROR] hdbscan not installed. Run: pip install hdbscan", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Running HDBSCAN on E. coli {X_3d.shape[1]}D UMAP: "
          f"min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_3d)
    probs = clusterer.probabilities_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    print(f"[INFO] E. coli HDBSCAN: {n_clusters} cluster(s), noise fraction={noise_frac:.3f}")
    return labels, probs


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identify E. coli admissions
    print("[INFO] Identifying E. coli admissions...")
    ecoli_ids = identify_ecoli_admissions(
        Path(args.feature_csv), args.ecoli_organism_col, args.ecoli_string
    )

    # 2. Load E. coli subset embeddings
    print("[INFO] Loading E. coli subset embeddings...")
    X_ecoli, meta_ecoli_df = load_ecoli_embeddings(
        Path(args.metadata_csv), Path(args.embeddings_dir), ecoli_ids
    )

    # Save subset metadata
    meta_ecoli_df.to_csv(out_dir / "ecoli_subset_metadata.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'ecoli_subset_metadata.csv'}")

    # 3a. 3D UMAP for HDBSCAN clustering
    umap_3d = run_umap(
        X_ecoli,
        n_components=3,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
        label="3D for HDBSCAN clustering",
    )

    # 3b. 2D UMAP for visualization only (not used for cluster assignment)
    umap_2d = run_umap(
        X_ecoli,
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
        label="2D for visualization only (not used for clustering)",
    )

    # Save 3D UMAP coordinates
    umap_3d_df = pd.DataFrame({
        "admission_ID": meta_ecoli_df["admission_ID"].values,
        "umap_c1": umap_3d[:, 0],
        "umap_c2": umap_3d[:, 1],
        "umap_c3": umap_3d[:, 2],
    })
    umap_3d_df.to_csv(out_dir / "ecoli_umap_3d_for_clustering.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'ecoli_umap_3d_for_clustering.csv'}")

    # Save 2D visualization UMAP coordinates
    umap_2d_df = pd.DataFrame({
        "admission_ID": meta_ecoli_df["admission_ID"].values,
        "umap_x": umap_2d[:, 0],
        "umap_y": umap_2d[:, 1],
    })
    umap_2d_df.to_csv(out_dir / "ecoli_umap_2d_for_visualization.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'ecoli_umap_2d_for_visualization.csv'}")

    # 4. HDBSCAN on 3D UMAP coordinates
    labels, probs = run_hdbscan(umap_3d, args.hdbscan_min_cluster_size, args.hdbscan_min_samples)

    # Save HDBSCAN labels
    hdbscan_df = pd.DataFrame({
        "admission_ID": meta_ecoli_df["admission_ID"].values,
        "cluster_label": labels,
        "cluster_probability": probs,
    })
    hdbscan_df.to_csv(out_dir / "ecoli_hdbscan_labels.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'ecoli_hdbscan_labels.csv'}")

    # 5. Cluster summary
    summary_df = (
        hdbscan_df.groupby("cluster_label")
        .agg(n=("admission_ID", "count"), mean_probability=("cluster_probability", "mean"))
        .reset_index()
    )
    summary_df.to_csv(out_dir / "ecoli_cluster_summary.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'ecoli_cluster_summary.csv'}")

    print(f"\n[INFO] Done. All E. coli analysis outputs in {out_dir}")


if __name__ == "__main__":
    main()
