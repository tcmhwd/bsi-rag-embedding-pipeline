#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
phenotype_interpretability.py

Characterizes identified BSI phenotype clusters using SHAP (SHapley Additive
exPlanations) values and permutation feature importance (PFI). Supports both
structured-variable interpretation (what clinical features drive cluster membership)
and narrative-level analysis.

Steps:
  1. Train a multiclass classifier (LightGBM) to predict cluster membership from
     structured clinical features.
  2. Compute SHAP values for each cluster using TreeExplainer.
  3. Compute permutation feature importance.
  4. Save per-cluster SHAP summaries and importance rankings.

Outputs (in --out-dir):
  - shap_values.npy               (n_samples, n_features, n_clusters)
  - shap_base_values.npy          (n_clusters,)
  - pfi_results.csv               (feature, mean_decrease_auroc, std)
  - shap_summary_per_cluster.csv  (mean |SHAP| per feature per cluster)
  - feature_names.json

Usage:
  python phenotype_interpretability.py \\
    --feature-csv ./data/admission_level_base.csv \\
    --cluster-labels ./clustering_out/hdbscan_labels.csv \\
    --out-dir ./interpretability_out \\
    --n-background 500 \\
    --n-pfi-repeats 10 \\
    --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

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


def parse_args():
    p = argparse.ArgumentParser(description="SHAP and PFI interpretability for BSI phenotype clusters.")
    p.add_argument("--feature-csv", type=str, required=True)
    p.add_argument("--cluster-labels", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--n-background", type=int, default=500,
                   help="Background samples for SHAP TreeExplainer (default: 500)")
    p.add_argument("--n-pfi-repeats", type=int, default=10,
                   help="Permutation repeats for PFI (default: 10)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(feature_csv: Path, cluster_csv: Path):
    feat_df = pd.read_csv(feature_csv, low_memory=False)
    feat_df["admission_ID"] = pd.to_numeric(feat_df["admission_ID"], errors="coerce")
    cluster_df = pd.read_csv(cluster_csv)
    cluster_df["admission_ID"] = pd.to_numeric(cluster_df["admission_ID"], errors="coerce")

    df = feat_df.merge(cluster_df[["admission_ID", "cluster_label"]], on="admission_ID", how="inner")
    df = df[df["cluster_label"] != -1].reset_index(drop=True)

    available = [c for c in STRUCTURED_FEATURES if c in df.columns]
    X = df[available].fillna(df[available].median())
    y = df["cluster_label"].astype(int)
    return X, y, available


def train_lgbm(X: pd.DataFrame, y: pd.Series, seed: int):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed. Run: pip install lightgbm", file=sys.stderr)
        sys.exit(1)

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)
    return model


def compute_shap(model, X: pd.DataFrame, n_background: int, seed: int):
    try:
        import shap
    except ImportError:
        print("[ERROR] shap not installed. Run: pip install shap", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X), size=min(n_background, len(X)), replace=False)
    background = X.iloc[bg_idx]

    explainer = shap.TreeExplainer(model, data=background, model_output="raw")
    shap_values = explainer.shap_values(X)
    base_values = explainer.expected_value

    return np.array(shap_values), np.array(base_values)


def compute_pfi(model, X: pd.DataFrame, y: pd.Series, n_repeats: int, seed: int):
    from sklearn.metrics import roc_auc_score

    def scorer(estimator, X_, y_):
        prob = estimator.predict_proba(X_)
        try:
            return roc_auc_score(y_, prob, multi_class="ovr", average="macro")
        except Exception:
            return 0.0

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring=scorer, n_jobs=-1
    )
    pfi_df = pd.DataFrame({
        "feature": X.columns,
        "mean_decrease_auroc": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("mean_decrease_auroc", ascending=False)
    return pfi_df


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    X, y, feature_names = load_data(Path(args.feature_csv), Path(args.cluster_labels))
    n_clusters = y.nunique()
    print(f"[INFO] n={len(y)}, clusters={n_clusters}, features={len(feature_names)}")

    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print("[INFO] Training LightGBM classifier on cluster labels...")
    model = train_lgbm(X, y, args.seed)

    print(f"[INFO] Computing SHAP values (background n={args.n_background})...")
    shap_values, base_values = compute_shap(model, X, args.n_background, args.seed)

    # shap_values shape: (n_classes, n_samples, n_features) -> transpose to (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        shap_out = np.transpose(shap_values, (1, 2, 0))
    else:
        shap_out = shap_values
    np.save(out_dir / "shap_values.npy", shap_out.astype(np.float32))
    np.save(out_dir / "shap_base_values.npy", base_values.astype(np.float32))
    print(f"[INFO] SHAP values saved: {shap_out.shape}")

    # Per-cluster mean |SHAP|
    cluster_ids = sorted(y.unique())
    records = []
    for ci, cluster in enumerate(cluster_ids):
        mask = (y == cluster).values
        cluster_shap = shap_out[mask, :, ci] if shap_out.ndim == 3 else shap_out[mask, :]
        mean_abs = np.abs(cluster_shap).mean(axis=0)
        for fi, feat in enumerate(feature_names):
            records.append({"cluster": cluster, "feature": feat, "mean_abs_shap": float(mean_abs[fi])})
    shap_summary = pd.DataFrame(records)
    shap_summary.to_csv(out_dir / "shap_summary_per_cluster.csv", index=False)

    print(f"[INFO] Computing PFI (n_repeats={args.n_pfi_repeats})...")
    pfi_df = compute_pfi(model, X, y, args.n_pfi_repeats, args.seed)
    pfi_df.to_csv(out_dir / "pfi_results.csv", index=False)

    print(f"[INFO] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
