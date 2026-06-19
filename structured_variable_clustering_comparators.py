#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structured-variable clustering comparator analyses for BSI phenotyping.

Runs two structured-variable comparator clustering pipelines to compare with
embedding-derived phenotype labels:
  1. PCA + k-means clustering
  2. Structured-variable UMAP (3D) + HDBSCAN clustering
     (with separate 2D UMAP for visualization)

These are descriptive comparator analyses, not prognostic prediction models.
Cluster concordance with embedding-derived phenotypes is assessed using ARI, NMI,
and contingency tables.

Outputs (in --out-dir):
  - pca_kmeans_labels.csv                  (admission_ID, kmeans_k{k}_label per k)
  - structured_umap_hdbscan_labels.csv     (admission_ID, cluster_label, cluster_probability)
  - structured_umap_2d_for_visualization.csv (admission_ID, umap_x, umap_y)
  - structured_umap_3d_for_clustering.csv  (admission_ID, umap_c1, umap_c2, umap_c3)
  - concordance_results.csv                (comparison_pair, ari, nmi)
  - contingency_tables/                    (one CSV per comparison pair)

Usage:
  python structured_variable_clustering_comparators.py \\
    --feature-csv ./data/admission_level_base.csv \\
    --cluster-labels ./clustering_out/hdbscan_labels.csv \\
    --out-dir ./comparators_out \\
    --n-pca-components 50 \\
    --kmeans-k-range 2,3,4,5,6,7,8 \\
    --umap-n-neighbors 15 \\
    --umap-min-dist 0.1 \\
    --hdbscan-min-cluster-size 30 \\
    --hdbscan-min-samples 5 \\
    --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

STRUCTURED_FEATURES = [
    "age_at_test", "sex",
    "charlson_score",
    "SBP", "RR", "PR", "Body_temperature", "GCS_2",
    "WBC_Count", "CRP", "Creatinine", "T_Bilirubin", "PLT_Count", "Albumin", "Hemoglobin",
    "icu", "Vasopressin_any",
    "sofa_score_2", "quick_sofa_score_2",
    "BSI_episode_3GCR", "BSI_episode_CarbaR", "BSI_episode_MRSA", "BSI_episode_VRE",
    "is_empiric_tt_excessivelylarge",
    "VRE_colonization", "CRE_colonization",
]

# Binary features — kept as-is after median imputation (already 0/1 coded)
BINARY_FEATURES = [
    "sex", "icu", "Vasopressin_any",
    "BSI_episode_3GCR", "BSI_episode_CarbaR", "BSI_episode_MRSA", "BSI_episode_VRE",
    "is_empiric_tt_excessivelylarge",
    "VRE_colonization", "CRE_colonization",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Structured-variable comparator clustering analyses for BSI phenotyping. "
            "Runs PCA+k-means and structured UMAP+HDBSCAN, then assesses concordance "
            "with embedding-derived phenotype labels. "
            "These are descriptive comparator analyses, not prognostic prediction models."
        )
    )
    p.add_argument("--feature-csv", type=str, required=True,
                   help="admission_level_base.csv with structured clinical features")
    p.add_argument("--cluster-labels", type=str, required=True,
                   help="hdbscan_labels.csv with embedding-derived cluster_label column")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write all outputs")
    p.add_argument("--n-pca-components", type=int, default=50,
                   help="Number of PCA components before k-means (default: 50)")
    p.add_argument("--kmeans-k-range", type=str, default="2,3,4,5,6,7,8",
                   help="Comma-separated k values for k-means (default: 2,3,4,5,6,7,8)")
    p.add_argument("--umap-n-neighbors", type=int, default=15,
                   help="UMAP n_neighbors for structured-variable UMAP (default: 15)")
    p.add_argument("--umap-min-dist", type=float, default=0.1,
                   help="UMAP min_dist for structured-variable UMAP (default: 0.1)")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=30,
                   help="HDBSCAN min_cluster_size (default: 30)")
    p.add_argument("--hdbscan-min-samples", type=int, default=5,
                   help="HDBSCAN min_samples (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def preprocess_features(df: pd.DataFrame, feature_cols: list):
    """Impute missing values with median and scale continuous features.

    Binary features (0/1 coded) are imputed with median but not rescaled.
    Continuous features are standardized with StandardScaler.

    Returns
    -------
    X_scaled : np.ndarray
        Preprocessed feature matrix (n_samples, n_features).
    feature_names_used : list of str
        Ordered list of feature names corresponding to columns of X_scaled.
    """
    available = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(available)
    if missing_cols:
        print(f"[WARN] Missing feature columns (skipped): {sorted(missing_cols)}")

    X_raw = df[available].copy()

    # Impute all features with median
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    # Scale continuous features only; keep binary features as-is
    continuous_idx = [i for i, c in enumerate(available) if c not in BINARY_FEATURES]
    binary_idx = [i for i, c in enumerate(available) if c in BINARY_FEATURES]

    if continuous_idx:
        scaler = StandardScaler()
        X_imp[:, continuous_idx] = scaler.fit_transform(X_imp[:, continuous_idx])

    return X_imp, available


def run_pca_kmeans(X: np.ndarray, n_pca_components: int, k_range: list, seed: int) -> dict:
    """Run PCA then k-means for each k in k_range.

    Returns
    -------
    dict : {k: labels_array} for each k in k_range.
    """
    n_components = min(n_pca_components, X.shape[0], X.shape[1])
    print(f"[INFO] PCA: n_components={n_components}")
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    print(f"[INFO] PCA variance explained (cumulative top-{n_components}): "
          f"{pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    results = {}
    for k in k_range:
        print(f"[INFO] k-means k={k}...")
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_pca)
        results[k] = labels
        print(f"[INFO]   k={k}: cluster sizes = {dict(zip(*np.unique(labels, return_counts=True)))}")
    return results


def run_structured_umap_hdbscan(X: np.ndarray, umap_params: dict,
                                 hdbscan_params: dict, seed: int):
    """Run structured-variable UMAP (3D for clustering, 2D for visualization) + HDBSCAN.

    Returns
    -------
    labels_3d : np.ndarray
        HDBSCAN labels derived from 3D UMAP coordinates.
    umap_2d_coords : np.ndarray
        2D UMAP coordinates for visualization only. NOT used for cluster assignment.
    umap_3d_coords : np.ndarray
        3D UMAP coordinates used for HDBSCAN clustering.
    probabilities : np.ndarray
        HDBSCAN cluster membership probabilities.
    """
    try:
        import umap as umap_lib
    except ImportError:
        print("[ERROR] umap-learn not installed. Run: pip install umap-learn", file=sys.stderr)
        sys.exit(1)
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        print("[ERROR] hdbscan not installed. Run: pip install hdbscan", file=sys.stderr)
        sys.exit(1)

    # 3D UMAP for HDBSCAN clustering input
    print(f"[INFO] Structured-variable UMAP (3D for clustering): {umap_params}")
    reducer_3d = umap_lib.UMAP(
        n_components=3,
        n_neighbors=umap_params["n_neighbors"],
        min_dist=umap_params["min_dist"],
        metric=umap_params.get("metric", "euclidean"),
        random_state=seed,
        verbose=True,
    )
    umap_3d_coords = reducer_3d.fit_transform(X)
    print(f"[INFO] Structured UMAP 3D complete: {umap_3d_coords.shape}")

    # 2D UMAP for visualization only — NOT used for cluster assignment
    print(f"[INFO] Structured-variable UMAP (2D for visualization only): {umap_params}")
    reducer_2d = umap_lib.UMAP(
        n_components=2,
        n_neighbors=umap_params["n_neighbors"],
        min_dist=umap_params["min_dist"],
        metric=umap_params.get("metric", "euclidean"),
        random_state=seed,
        verbose=True,
    )
    umap_2d_coords = reducer_2d.fit_transform(X)
    print(f"[INFO] Structured UMAP 2D (visualization only) complete: {umap_2d_coords.shape}")

    # HDBSCAN on 3D UMAP coordinates only
    print(f"[INFO] HDBSCAN on 3D structured UMAP: {hdbscan_params}")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=hdbscan_params["min_cluster_size"],
        min_samples=hdbscan_params["min_samples"],
        prediction_data=True,
    )
    labels_3d = clusterer.fit_predict(umap_3d_coords)
    probabilities = clusterer.probabilities_
    n_clusters = len(set(labels_3d)) - (1 if -1 in labels_3d else 0)
    noise_frac = (labels_3d == -1).mean()
    print(f"[INFO] Structured HDBSCAN: {n_clusters} cluster(s), noise fraction={noise_frac:.3f}")

    return labels_3d, umap_2d_coords, umap_3d_coords, probabilities


def compute_concordance(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Compute ARI and NMI between two label arrays.

    Noise points (label == -1) are excluded from concordance computation.

    Returns
    -------
    dict with keys: ari, nmi, n_compared
    """
    mask = (labels_a != -1) & (labels_b != -1)
    a = labels_a[mask]
    b = labels_b[mask]
    if len(a) == 0:
        return {"ari": np.nan, "nmi": np.nan, "n_compared": 0}
    ari = adjusted_rand_score(a, b)
    nmi = normalized_mutual_info_score(a, b, average_method="arithmetic")
    return {"ari": ari, "nmi": nmi, "n_compared": int(mask.sum())}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contingency_dir = out_dir / "contingency_tables"
    contingency_dir.mkdir(parents=True, exist_ok=True)

    # Parse k_range
    k_range = [int(k.strip()) for k in args.kmeans_k_range.split(",") if k.strip()]

    # Load data
    print("[INFO] Loading feature CSV and embedding-derived cluster labels...")
    feat_df = pd.read_csv(args.feature_csv, low_memory=False)
    feat_df["admission_ID"] = pd.to_numeric(feat_df["admission_ID"], errors="coerce")

    cluster_df = pd.read_csv(args.cluster_labels)
    cluster_df["admission_ID"] = pd.to_numeric(cluster_df["admission_ID"], errors="coerce")

    df = feat_df.merge(cluster_df[["admission_ID", "cluster_label"]], on="admission_ID", how="inner")
    print(f"[INFO] Merged: n={len(df)} admissions")

    embedding_labels = df["cluster_label"].values
    admission_ids = df["admission_ID"].values

    # Preprocess features
    print("[INFO] Preprocessing structured features...")
    X_scaled, feature_names_used = preprocess_features(df, STRUCTURED_FEATURES)
    print(f"[INFO] Feature matrix shape: {X_scaled.shape}, features used: {len(feature_names_used)}")

    concordance_records = []

    # --- PCA + k-means ---
    print("\n[INFO] Running PCA + k-means comparator...")
    kmeans_results = run_pca_kmeans(X_scaled, args.n_pca_components, k_range, args.seed)

    kmeans_label_df = pd.DataFrame({"admission_ID": admission_ids})
    for k, labels in kmeans_results.items():
        col = f"kmeans_k{k}_label"
        kmeans_label_df[col] = labels

        # Concordance with embedding-derived labels
        conc = compute_concordance(embedding_labels, labels)
        concordance_records.append({
            "comparison_pair": f"embedding_vs_kmeans_k{k}",
            "ari": conc["ari"],
            "nmi": conc["nmi"],
            "n_compared": conc["n_compared"],
        })
        print(f"[INFO] Concordance embedding vs k-means k={k}: ARI={conc['ari']:.3f}, NMI={conc['nmi']:.3f}")

        # Contingency table
        ct = pd.crosstab(embedding_labels, labels,
                         rownames=["embedding_cluster"], colnames=[f"kmeans_k{k}"])
        ct.to_csv(contingency_dir / f"contingency_embedding_vs_kmeans_k{k}.csv")

    kmeans_label_df.to_csv(out_dir / "pca_kmeans_labels.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'pca_kmeans_labels.csv'}")

    # --- Structured UMAP + HDBSCAN ---
    print("\n[INFO] Running structured-variable UMAP + HDBSCAN comparator...")
    umap_params = {
        "n_neighbors": args.umap_n_neighbors,
        "min_dist": args.umap_min_dist,
        "metric": "euclidean",
    }
    hdbscan_params = {
        "min_cluster_size": args.hdbscan_min_cluster_size,
        "min_samples": args.hdbscan_min_samples,
    }
    struct_labels, umap_2d_coords, umap_3d_coords, struct_probs = run_structured_umap_hdbscan(
        X_scaled, umap_params, hdbscan_params, args.seed
    )

    # Save structured UMAP 3D coordinates
    struct_3d_df = pd.DataFrame({
        "admission_ID": admission_ids,
        "umap_c1": umap_3d_coords[:, 0],
        "umap_c2": umap_3d_coords[:, 1],
        "umap_c3": umap_3d_coords[:, 2],
    })
    struct_3d_df.to_csv(out_dir / "structured_umap_3d_for_clustering.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'structured_umap_3d_for_clustering.csv'}")

    # Save structured UMAP 2D visualization coordinates
    struct_2d_df = pd.DataFrame({
        "admission_ID": admission_ids,
        "umap_x": umap_2d_coords[:, 0],
        "umap_y": umap_2d_coords[:, 1],
    })
    struct_2d_df.to_csv(out_dir / "structured_umap_2d_for_visualization.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'structured_umap_2d_for_visualization.csv'}")

    # Save structured UMAP+HDBSCAN labels
    struct_hdbscan_df = pd.DataFrame({
        "admission_ID": admission_ids,
        "cluster_label": struct_labels,
        "cluster_probability": struct_probs,
    })
    struct_hdbscan_df.to_csv(out_dir / "structured_umap_hdbscan_labels.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'structured_umap_hdbscan_labels.csv'}")

    # Concordance: embedding vs structured UMAP+HDBSCAN
    conc_struct = compute_concordance(embedding_labels, struct_labels)
    concordance_records.append({
        "comparison_pair": "embedding_vs_structured_umap_hdbscan",
        "ari": conc_struct["ari"],
        "nmi": conc_struct["nmi"],
        "n_compared": conc_struct["n_compared"],
    })
    print(f"[INFO] Concordance embedding vs structured UMAP+HDBSCAN: "
          f"ARI={conc_struct['ari']:.3f}, NMI={conc_struct['nmi']:.3f}")

    ct_struct = pd.crosstab(embedding_labels, struct_labels,
                             rownames=["embedding_cluster"],
                             colnames=["structured_umap_hdbscan"])
    ct_struct.to_csv(contingency_dir / "contingency_embedding_vs_structured_umap_hdbscan.csv")

    # Save concordance results
    concordance_df = pd.DataFrame(concordance_records)
    concordance_df.to_csv(out_dir / "concordance_results.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'concordance_results.csv'}")

    print(f"\n[INFO] Done. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
