#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cluster_embeddings_umap_hdbscan.py

Loads narrative embeddings, applies UMAP dimensionality reduction in two separate
roles, then clusters with HDBSCAN to identify BSI phenotype clusters.

UMAP roles:
  - UMAP is used in two separate roles in this pipeline.
  - HDBSCAN clustering is performed on a 3-dimensional UMAP representation
    (default --umap-n-components-cluster 3).
  - A separate 2-dimensional UMAP projection is generated only for visualization
    (default --umap-n-components-viz 2).
  - The displayed 2-dimensional UMAP projection is NOT used for cluster assignment.

Steps:
  1. Load sharded .npy embeddings and metadata CSV.
  2. Run UMAP with n_components = --umap-n-components-cluster (default 3) for
     HDBSCAN clustering input.
  3. Run a SEPARATE UMAP with n_components = --umap-n-components-viz (default 2)
     for visualization only.
  4. HDBSCAN clusters the 3D UMAP coordinates ONLY.
  5. Save UMAP coordinates, cluster labels, parameters, and cluster summary.

Outputs (in --out-dir):
  - umap_3d_for_clustering.csv      (admission_ID, umap_c1, umap_c2, umap_c3)
  - umap_2d_for_visualization.csv   (admission_ID, umap_x, umap_y)
  - hdbscan_labels.csv              (admission_ID, patient_ID [if available],
                                     cluster_label, cluster_probability)
  - clustering_parameters.json      (all hyperparameters)
  - cluster_summary.csv             (cluster-level size and mean probability)

Usage:
  python cluster_embeddings_umap_hdbscan.py \\
    --embeddings-dir ./embeddings_out \\
    --metadata-csv ./embeddings_out/narrative_embeddings_metadata.csv \\
    --out-dir ./clustering_out \\
    --outcome-csv ./data/admission_level_base.csv \\
    --umap-n-components-cluster 3 \\
    --umap-n-components-viz 2 \\
    --umap-n-neighbors 15 \\
    --umap-min-dist 0.1 \\
    --umap-metric cosine \\
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
    p = argparse.ArgumentParser(
        description=(
            "UMAP + HDBSCAN clustering of BSI narrative embeddings. "
            "UMAP is run twice: once for HDBSCAN clustering input (3D by default), "
            "and once for visualization only (2D by default). "
            "Cluster assignment uses only the 3D UMAP coordinates."
        )
    )
    p.add_argument("--embeddings-dir", type=str, required=True,
                   help="Directory containing embeddings_shards/ subdirectory")
    p.add_argument("--metadata-csv", type=str, required=True,
                   help="narrative_embeddings_metadata.csv with admission_ID column")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write all outputs")
    p.add_argument("--outcome-csv", type=str, default=None,
                   help="Optional: admission_level_base.csv to merge clinical variables onto output")
    p.add_argument("--umap-n-components-cluster", type=int, default=3,
                   help="UMAP dimensions used as input to HDBSCAN clustering (default: 3)")
    p.add_argument("--umap-n-components-viz", type=int, default=2,
                   help="UMAP dimensions for visualization only; NOT used for cluster assignment (default: 2)")
    p.add_argument("--umap-n-neighbors", type=int, default=15,
                   help="UMAP n_neighbors parameter (default: 15)")
    p.add_argument("--umap-min-dist", type=float, default=0.1,
                   help="UMAP min_dist parameter (default: 0.1)")
    p.add_argument("--umap-metric", type=str, default="cosine",
                   help="UMAP distance metric (default: cosine)")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=50,
                   help="HDBSCAN min_cluster_size (default: 50)")
    p.add_argument("--hdbscan-min-samples", type=int, default=10,
                   help="HDBSCAN min_samples (default: 10)")
    p.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=0.0,
                   help="HDBSCAN cluster_selection_epsilon (default: 0.0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for UMAP reproducibility (default: 42)")
    return p.parse_args()


def load_embeddings(embeddings_dir: Path, metadata_csv: Path):
    """Load sharded .npy embedding files and metadata CSV.

    Expects shards in embeddings_dir/embeddings_shards/shard_*.npy.
    Returns X (float32 array) and meta_df (DataFrame with at least admission_ID).
    """
    meta_df = pd.read_csv(metadata_csv)
    shard_dir = embeddings_dir / "embeddings_shards"
    shard_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not shard_paths:
        print(f"[ERROR] No shard files found in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    arrays = [np.load(sp) for sp in shard_paths]
    X = np.concatenate(arrays, axis=0).astype(np.float32)
    print(f"[INFO] Loaded embeddings: shape={X.shape} from {len(shard_paths)} shard(s)")

    if len(meta_df) != X.shape[0]:
        print(
            f"[WARN] Metadata rows ({len(meta_df)}) != embedding rows ({X.shape[0]}). "
            "Ensure metadata and embedding shards are aligned.",
            file=sys.stderr,
        )

    return X, meta_df


def run_umap(X: np.ndarray, n_components: int, n_neighbors: int, min_dist: float,
             metric: str, seed: int, label: str) -> np.ndarray:
    """Run a single UMAP model and return the embedding array.

    Parameters
    ----------
    label : str
        Descriptive label printed in log output (e.g. '3D clustering' or '2D visualization').
    """
    try:
        import umap
    except ImportError:
        print("[ERROR] umap-learn not installed. Run: pip install umap-learn", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Running UMAP [{label}]: n_components={n_components}, "
          f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}, seed={seed}")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
        verbose=True,
    )
    embedding = reducer.fit_transform(X)
    print(f"[INFO] UMAP [{label}] complete: output shape={embedding.shape}")
    return embedding


def run_hdbscan(X_3d: np.ndarray, min_cluster_size: int, min_samples: int,
                epsilon: float):
    """Run HDBSCAN on 3D UMAP coordinates.

    Parameters
    ----------
    X_3d : np.ndarray
        The 3-dimensional UMAP coordinates. HDBSCAN is applied to these coordinates only.
    """
    try:
        import hdbscan
    except ImportError:
        print("[ERROR] hdbscan not installed. Run: pip install hdbscan", file=sys.stderr)
        sys.exit(1)

    # HDBSCAN is applied to the 3D UMAP coordinates only.
    print(f"[INFO] Running HDBSCAN on {X_3d.shape[1]}D UMAP coordinates: "
          f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_3d)
    probs = clusterer.probabilities_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    print(f"[INFO] HDBSCAN complete: {n_clusters} cluster(s), noise fraction={noise_frac:.3f}")
    return labels, probs


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    # 1. Load embeddings
    X, meta_df = load_embeddings(embeddings_dir, Path(args.metadata_csv))

    # 2. Run UMAP for HDBSCAN clustering input (default: 3D)
    umap_clustering_coords = run_umap(
        X,
        n_components=args.umap_n_components_cluster,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
        label=f"{args.umap_n_components_cluster}D for HDBSCAN clustering",
    )

    # 3. Run SEPARATE UMAP for visualization only (default: 2D)
    # 2D UMAP is for visualization only and is not used for cluster assignment.
    umap_viz_coords = run_umap(
        X,
        n_components=args.umap_n_components_viz,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
        label=f"{args.umap_n_components_viz}D for visualization only (not used for clustering)",
    )

    # 4. Run HDBSCAN on clustering UMAP coordinates ONLY
    labels, probs = run_hdbscan(
        umap_clustering_coords,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        epsilon=args.hdbscan_cluster_selection_epsilon,
    )

    # 5. Save outputs

    # umap_3d_for_clustering.csv
    n_cluster_dims = args.umap_n_components_cluster
    umap_clustering_df = meta_df[["admission_ID"]].copy()
    for i in range(n_cluster_dims):
        umap_clustering_df[f"umap_c{i+1}"] = umap_clustering_coords[:, i]
    umap_clustering_df.to_csv(out_dir / "umap_3d_for_clustering.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'umap_3d_for_clustering.csv'}")

    # umap_2d_for_visualization.csv
    # 2D UMAP is for visualization only and is not used for cluster assignment.
    umap_viz_df = meta_df[["admission_ID"]].copy()
    umap_viz_df["umap_x"] = umap_viz_coords[:, 0]
    umap_viz_df["umap_y"] = umap_viz_coords[:, 1]
    umap_viz_df.to_csv(out_dir / "umap_2d_for_visualization.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'umap_2d_for_visualization.csv'}")

    # hdbscan_labels.csv
    if "patient_ID" in meta_df.columns:
        label_df = meta_df[["admission_ID", "patient_ID"]].copy()
    else:
        label_df = meta_df[["admission_ID"]].copy()
    label_df["cluster_label"] = labels
    label_df["cluster_probability"] = probs

    if args.outcome_csv:
        outcome_df = pd.read_csv(args.outcome_csv, low_memory=False)
        outcome_df["admission_ID"] = pd.to_numeric(outcome_df["admission_ID"], errors="coerce")
        label_df = label_df.merge(outcome_df, on="admission_ID", how="left")

    label_df.to_csv(out_dir / "hdbscan_labels.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'hdbscan_labels.csv'}")

    # clustering_parameters.json
    params = {
        "umap_n_components_cluster": args.umap_n_components_cluster,
        "umap_n_components_viz": args.umap_n_components_viz,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "hdbscan_cluster_selection_epsilon": args.hdbscan_cluster_selection_epsilon,
        "seed": args.seed,
        "note_umap_roles": (
            "umap_n_components_cluster: dimensions used for HDBSCAN clustering input. "
            "umap_n_components_viz: dimensions used for visualization only, "
            "NOT used for cluster assignment."
        ),
    }
    params_path = out_dir / "clustering_parameters.json"
    params_path.write_text(json.dumps(params, indent=2))
    print(f"[INFO] Saved: {params_path}")

    # cluster_summary.csv
    summary_df = (
        label_df.groupby("cluster_label")
        .agg(n=("admission_ID", "count"), mean_probability=("cluster_probability", "mean"))
        .reset_index()
    )
    summary_df.to_csv(out_dir / "cluster_summary.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'cluster_summary.csv'}")

    print(f"[INFO] Done. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
