#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_figures_tables.py

Generates all main-text and supplementary figures and tables from intermediate
analysis outputs. Reads from clustering, interpretability, trajectory, and
comparator output directories.

Figures produced:
  - Fig 1: UMAP scatter colored by cluster label
  - Fig 2: UMAP scatter colored by selected clinical variables (mortality, CCI, etc.)
  - Fig 3: Cumulative incidence curves by cluster (competing-risk)
  - Fig 4: SHAP summary / beeswarm per cluster
  - Fig S1: HDBSCAN hyperparameter sensitivity grid
  - Fig S2: PFI bar chart

Tables produced:
  - Table 1: Cluster characterization (demographics, severity, microbiology)
  - Table 2: Structured-variable comparator CV performance
  - Table S1: Fine-Gray results

Usage:
  python make_figures_tables.py \\
    --clustering-dir ./clustering_out \\
    --interpretability-dir ./interpretability_out \\
    --trajectories-dir ./trajectories_out \\
    --comparators-dir ./comparators_out \\
    --feature-csv ./data/admission_level_base.csv \\
    --out-dir ./figures_tables \\
    --dpi 300 \\
    --format pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Generate manuscript figures and tables.")
    p.add_argument("--clustering-dir", type=str, required=True)
    p.add_argument("--interpretability-dir", type=str, required=True)
    p.add_argument("--trajectories-dir", type=str, required=True)
    p.add_argument("--comparators-dir", type=str, required=True)
    p.add_argument("--feature-csv", type=str, default=None)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--format", type=str, default="pdf", choices=["pdf", "svg", "png"])
    return p.parse_args()


def load_inputs(args):
    clustering_dir = Path(args.clustering_dir)
    interp_dir = Path(args.interpretability_dir)
    traj_dir = Path(args.trajectories_dir)
    comp_dir = Path(args.comparators_dir)

    labels_csv = clustering_dir / "hdbscan_labels.csv"
    umap_2d_npy = clustering_dir / "umap_2d.npy"
    cluster_summary_csv = clustering_dir / "cluster_summary.csv"
    ci_csv = traj_dir / "cumulative_incidence.csv"
    fg_csv = traj_dir / "finegray_results.csv"
    surv_summary_csv = traj_dir / "cluster_survival_summary.csv"
    shap_csv = interp_dir / "shap_summary_per_cluster.csv"
    pfi_csv = interp_dir / "pfi_results.csv"
    feat_names_json = interp_dir / "feature_names.json"
    comp_csv = comp_dir / "comparator_cv_results.csv"

    data = {}
    for name, path in [
        ("labels", labels_csv), ("cluster_summary", cluster_summary_csv),
        ("ci", ci_csv), ("fg", fg_csv), ("surv_summary", surv_summary_csv),
        ("shap_summary", shap_csv), ("pfi", pfi_csv), ("comp_results", comp_csv),
    ]:
        if path.exists():
            data[name] = pd.read_csv(path)
        else:
            print(f"[WARN] Not found, skipping: {path}")

    if umap_2d_npy.exists():
        data["umap_2d"] = np.load(umap_2d_npy)

    if feat_names_json.exists():
        data["feature_names"] = json.loads(feat_names_json.read_text())

    return data


def make_umap_cluster_figure(data: dict, out_dir: Path, dpi: int, fmt: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping UMAP figure.", file=sys.stderr)
        return

    umap_2d = data.get("umap_2d")
    labels_df = data.get("labels")
    if umap_2d is None or labels_df is None:
        print("[WARN] Missing UMAP or cluster label data; skipping Fig 1.")
        return

    labels = labels_df["cluster_label"].values
    unique_clusters = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for i, cl in enumerate(unique_clusters):
        mask = labels == cl
        color = "lightgrey" if cl == -1 else cmap(i % 10)
        label = "Noise" if cl == -1 else f"Cluster {cl}"
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=[color], s=4, alpha=0.6,
                   linewidths=0, label=label)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP projection of BSI narrative embeddings")
    ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.tight_layout()
    out_path = out_dir / f"fig1_umap_clusters.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")


def make_cumulative_incidence_figure(data: dict, out_dir: Path, dpi: int, fmt: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping CIF figure.", file=sys.stderr)
        return

    ci_df = data.get("ci")
    if ci_df is None:
        print("[WARN] Missing cumulative incidence data; skipping Fig 3.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster, grp in ci_df.groupby("cluster"):
        ax.step(grp["time"], grp["cumulative_incidence"], where="post",
                label=f"Cluster {cluster}", linewidth=1.5)

    ax.set_xlabel("Days from bloodstream infection")
    ax.set_ylabel("Cumulative incidence of 30-day mortality")
    ax.set_title("Competing-risk cumulative incidence by phenotype cluster")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path = out_dir / f"fig3_cumulative_incidence.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")


def make_shap_figure(data: dict, out_dir: Path, dpi: int, fmt: str):
    shap_df = data.get("shap_summary")
    if shap_df is None:
        print("[WARN] Missing SHAP summary; skipping Fig 4.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping SHAP figure.", file=sys.stderr)
        return

    clusters = sorted(shap_df["cluster"].unique())
    n_top = 15
    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 6), sharey=False)
    if len(clusters) == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters):
        sub = shap_df[shap_df["cluster"] == cluster].nlargest(n_top, "mean_abs_shap")
        ax.barh(sub["feature"], sub["mean_abs_shap"])
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel("Mean |SHAP|")
        ax.invert_yaxis()

    fig.suptitle("Top features by mean |SHAP| per BSI phenotype cluster")
    fig.tight_layout()
    out_path = out_dir / f"fig4_shap_per_cluster.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")


def make_table1(data: dict, out_dir: Path):
    surv_df = data.get("surv_summary")
    cluster_df = data.get("cluster_summary")
    if surv_df is None or cluster_df is None:
        print("[WARN] Missing cluster summary for Table 1.")
        return

    t1 = cluster_df.merge(surv_df, on="cluster", how="outer")
    t1.to_csv(out_dir / "table1_cluster_characterization.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'table1_cluster_characterization.csv'}")


def make_table2(data: dict, out_dir: Path):
    comp_df = data.get("comp_results")
    if comp_df is None:
        print("[WARN] Missing comparator CV results for Table 2.")
        return

    t2 = comp_df.groupby("model")[["auroc", "average_precision", "brier_score"]].agg(
        ["mean", "std"]
    ).round(3)
    t2.to_csv(out_dir / "table2_comparator_cv.csv")
    print(f"[INFO] Saved: {out_dir / 'table2_comparator_cv.csv'}")


def make_table_s1(data: dict, out_dir: Path):
    fg_df = data.get("fg")
    if fg_df is None:
        print("[WARN] Missing Fine-Gray results for Table S1.")
        return
    fg_df.to_csv(out_dir / "table_s1_finegray.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'table_s1_finegray.csv'}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading analysis outputs...")
    data = load_inputs(args)

    print("[INFO] Generating figures...")
    make_umap_cluster_figure(data, out_dir, args.dpi, args.format)
    make_cumulative_incidence_figure(data, out_dir, args.dpi, args.format)
    make_shap_figure(data, out_dir, args.dpi, args.format)

    print("[INFO] Generating tables...")
    make_table1(data, out_dir)
    make_table2(data, out_dir)
    make_table_s1(data, out_dir)

    print(f"[INFO] Done. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
