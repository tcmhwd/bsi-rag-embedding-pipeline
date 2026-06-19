#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cluster_embeddings_umap_hdbscan.py

Loads narrative embeddings, applies UMAP dimensionality reduction, then clusters
with HDBSCAN to identify BSI phenotype clusters.

Steps:
  1. Load sharded .npy embeddings and metadata CSV.
  2. Optionally merge with outcome CSV for downstream characterization.
  3. UMAP: reduce to 2D (visualization) and 3D (clustering input).
  4. HDBSCAN: cluster the UMAP-reduced embeddings.
  5. Save UMAP coordinates, cluster labels, and a cluster membership summary.

Outputs (in --out-dir):
  - umap_2d.npy                  (n_samples, 2)
  - umap_3d.npy                  (n_samples, 3)
  - hdbscan_labels.csv           (admission_ID, patient_ID, cluster_label, cluster_probability)
  - cluster_summary.csv          (cluster-level size and noise fraction)
  - umap_params.json             (hyperparameters used)

Usage:
  python cluster_embeddings_umap_hdbscan.py \\
    --embeddings-dir ./embeddings_out \\
    --metadata-csv ./embeddings_out/narrative_embeddings_metadata.csv \\
    --outcome-csv ./data/admission_level_base.csv \\
    --out-dir ./clustering_out \\
    --umap-n-neighbors 15 \\
    --umap-min-dist 0.1 \\
    --hdbscan-min-cluster-size 50 \\
    --hdbscan-min-samples 10 \\
    --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="UMAP + HDBSCAN clustering of BSI narrative embeddings.")
    p.add_argument("--embeddings-dir", type=str, required=True,
                   help="Directory containing embeddings_shards/ and narrative_embeddings_metadata.csv")
    p.add_argument("--metadata-csv", type=str, required=True)
    p.add_argument("--outcome-csv", type=str, default=None,
                   help="Optional: admission_level_base.csv to merge clinical variables onto cluster labels")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-n-components-cluster", type=int, default=3,
                   help="UMAP dimensions used as input to HDBSCAN (default: 3)")
    p.add_argument("--umap-metric", type=str, default="cosine")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=50)
    p.add_argument("--hdbscan-min-samples", type=int, default=10)
    p.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_embeddings(embeddings_dir: Path, metadata_csv: Path):
    meta_df = pd.read_csv(metadata_csv)
    shard_dir = embeddings_dir / "embeddings_shards"
    shard_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not shard_paths:
        print(f"[ERROR] No shard files found in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    arrays = [np.load(sp) for sp in shard_paths]
    X = np.concatenate(arrays, axis=0).astype(np.float32)
    print(f"[INFO] Loaded embeddings: {X.shape} from {len(shard_paths)} shards")
    if len(meta_df) != X.shape[0]:
        print(f"[WARN] Metadata rows ({len(meta_df)}) != embedding rows ({X.shape[0]})", file=sys.stderr)

    return X, meta_df


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, n_components: int,
             metric: str, seed: int):
    try:
        import umap
    except ImportError:
        print("[ERROR] umap-learn not installed. Run: pip install umap-learn", file=sys.stderr)
        sys.exit(1)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
        verbose=True,
    )
    embedding = reducer.fit_transform(X)
    print(f"[INFO] UMAP {n_components}D complete: {embedding.shape}")
    return embedding, reducer


def run_hdbscan(X_reduced: np.ndarray, min_cluster_size: int, min_samples: int,
                cluster_selection_epsilon: float):
    try:
        import hdbscan
    except ImportError:
        print("[ERROR] hdbscan not installed. Run: pip install hdbscan", file=sys.stderr)
        sys.exit(1)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_reduced)
    probs = clusterer.probabilities_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    print(f"[INFO] HDBSCAN: {n_clusters} clusters, noise fraction={noise_frac:.3f}")
    return labels, probs, clusterer


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    X, meta_df = load_embeddings(embeddings_dir, Path(args.metadata_csv))

    # UMAP 2D for visualization
    print("[INFO] Running UMAP 2D (visualization)...")
    umap_2d, _ = run_umap(X, args.umap_n_neighbors, args.umap_min_dist, 2,
                           args.umap_metric, args.seed)
    np.save(out_dir / "umap_2d.npy", umap_2d.astype(np.float32))

    # UMAP nD for clustering
    n_clust_dims = args.umap_n_components_cluster
    if n_clust_dims != 2:
        print(f"[INFO] Running UMAP {n_clust_dims}D (clustering input)...")
        umap_nd, _ = run_umap(X, args.umap_n_neighbors, args.umap_min_dist, n_clust_dims,
                               args.umap_metric, args.seed)
    else:
        umap_nd = umap_2d
    np.save(out_dir / f"umap_{n_clust_dims}d.npy", umap_nd.astype(np.float32))

    # HDBSCAN
    print("[INFO] Running HDBSCAN...")
    labels, probs, _ = run_hdbscan(
        umap_nd,
        args.hdbscan_min_cluster_size,
        args.hdbscan_min_samples,
        args.hdbscan_cluster_selection_epsilon,
    )

    label_df = meta_df[["admission_ID", "patient_ID"]].copy() if "patient_ID" in meta_df.columns else meta_df[["admission_ID"]].copy()
    label_df["cluster_label"] = labels
    label_df["cluster_probability"] = probs
    label_df["umap_x"] = umap_2d[:, 0]
    label_df["umap_y"] = umap_2d[:, 1]

    if args.outcome_csv:
        outcome_df = pd.read_csv(args.outcome_csv, low_memory=False)
        outcome_df["admission_ID"] = pd.to_numeric(outcome_df["admission_ID"], errors="coerce")
        label_df = label_df.merge(outcome_df, on="admission_ID", how="left")

    label_df.to_csv(out_dir / "hdbscan_labels.csv", index=False)
    print(f"[INFO] Cluster labels saved: {out_dir / 'hdbscan_labels.csv'}")

    # Cluster summary
    summary = (
        label_df.groupby("cluster_label")
        .agg(n=("admission_ID", "count"), mean_probability=("cluster_probability", "mean"))
        .reset_index()
    )
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    # Save hyperparameters
    params = {
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_n_components_cluster": n_clust_dims,
        "umap_metric": args.umap_metric,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "hdbscan_cluster_selection_epsilon": args.hdbscan_cluster_selection_epsilon,
        "seed": args.seed,
    }
    (out_dir / "umap_params.json").write_text(json.dumps(params, indent=2))

    print(f"[INFO] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
