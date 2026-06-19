#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
structured_variable_comparators.py

Trains and cross-validates structured-variable comparator models for BSI phenotype
assignment. Comparators are trained on the same admission-level feature set used to
construct case narratives, and evaluated using identical cross-validation splits to
allow fair comparison with embedding-based approaches.

Models:
  - Logistic Regression (L2-penalized)
  - Gradient Boosted Trees (LightGBM)

Each model is evaluated with stratified k-fold CV. Outputs include per-fold
performance metrics and out-of-fold predictions for calibration analysis.

Outputs (in --out-dir):
  - comparator_cv_results.csv     (per-fold AUROC, AP, Brier score per model)
  - oof_predictions.csv           (out-of-fold predicted probabilities)
  - feature_importance_lgbm.csv   (LightGBM gain-based feature importance)

Usage:
  python structured_variable_comparators.py \\
    --feature-csv ./data/admission_level_base.csv \\
    --cluster-labels ./clustering_out/hdbscan_labels.csv \\
    --out-dir ./comparators_out \\
    --cv-folds 5 \\
    --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
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


def parse_args():
    p = argparse.ArgumentParser(description="Structured-variable comparator analysis for BSI phenotyping.")
    p.add_argument("--feature-csv", type=str, required=True,
                   help="admission_level_base.csv with structured clinical features")
    p.add_argument("--cluster-labels", type=str, required=True,
                   help="hdbscan_labels.csv with cluster_label column (output of cluster_embeddings_umap_hdbscan.py)")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lgbm", action="store_true", default=True,
                   help="Include LightGBM comparator (default: on; requires lightgbm)")
    p.add_argument("--target-col", type=str, default="mortality_30d",
                   help="Binary outcome column for comparator evaluation (default: mortality_30d)")
    return p.parse_args()


def load_data(feature_csv: Path, cluster_csv: Path, target_col: str):
    feat_df = pd.read_csv(feature_csv, low_memory=False)
    feat_df["admission_ID"] = pd.to_numeric(feat_df["admission_ID"], errors="coerce")

    cluster_df = pd.read_csv(cluster_csv)
    cluster_df["admission_ID"] = pd.to_numeric(cluster_df["admission_ID"], errors="coerce")

    df = feat_df.merge(cluster_df[["admission_ID", "cluster_label"]], on="admission_ID", how="inner")
    df = df[df["cluster_label"] != -1]  # exclude HDBSCAN noise points

    available = [c for c in STRUCTURED_FEATURES if c in df.columns]
    missing = set(STRUCTURED_FEATURES) - set(available)
    if missing:
        print(f"[WARN] Missing feature columns (will be skipped): {missing}")

    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found. Available: {list(df.columns[:10])}", file=sys.stderr)
        sys.exit(1)

    X = df[available].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    df_out = df[["admission_ID", "cluster_label"]].copy()

    return X, y, df_out, available


def evaluate_fold(y_true, y_prob):
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def run_logistic(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, seed: int):
    X_imp = X.fillna(X.median())
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=seed, C=1.0)),
    ])
    fold_results = []
    oof_probs = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(cv.split(X_imp, y)):
        pipeline.fit(X_imp.iloc[tr], y.iloc[tr])
        prob = pipeline.predict_proba(X_imp.iloc[te])[:, 1]
        oof_probs[te] = prob
        metrics = evaluate_fold(y.iloc[te].values, prob)
        fold_results.append({"model": "logistic_regression", "fold": fold, **metrics})
        print(f"  [LR] fold={fold} AUROC={metrics['auroc']:.3f}")
    return fold_results, oof_probs


def run_lgbm(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, seed: int):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[WARN] lightgbm not installed; skipping LightGBM comparator.", file=sys.stderr)
        return [], np.full(len(y), np.nan), None

    fold_results = []
    oof_probs = np.full(len(y), np.nan)
    importances = []

    for fold, (tr, te) in enumerate(cv.split(X, y)):
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                    num_leaves=31, random_state=seed,
                                    n_jobs=-1, verbose=-1)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[te], y.iloc[te])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        prob = model.predict_proba(X.iloc[te])[:, 1]
        oof_probs[te] = prob
        metrics = evaluate_fold(y.iloc[te].values, prob)
        fold_results.append({"model": "lightgbm", "fold": fold, **metrics})
        importances.append(pd.Series(model.feature_importances_, index=X.columns))
        print(f"  [LGBM] fold={fold} AUROC={metrics['auroc']:.3f}")

    fi_df = pd.DataFrame(importances).mean().reset_index()
    fi_df.columns = ["feature", "mean_gain"]
    fi_df = fi_df.sort_values("mean_gain", ascending=False)
    return fold_results, oof_probs, fi_df


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, df_out, feature_cols = load_data(
        Path(args.feature_csv), Path(args.cluster_labels), args.target_col
    )
    valid = y.notna()
    X, y, df_out = X[valid], y[valid], df_out[valid]
    print(f"[INFO] n={len(y)}, pos_rate={y.mean():.3f}, features={len(feature_cols)}")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    all_results = []
    oof_df = df_out.copy()

    print("[INFO] Logistic Regression...")
    lr_results, lr_oof = run_logistic(X, y, cv, args.seed)
    all_results.extend(lr_results)
    oof_df["prob_logistic_regression"] = lr_oof

    if args.lgbm:
        print("[INFO] LightGBM...")
        lgbm_results, lgbm_oof, fi_df = run_lgbm(X, y, cv, args.seed)
        all_results.extend(lgbm_results)
        oof_df["prob_lightgbm"] = lgbm_oof
        if fi_df is not None:
            fi_df.to_csv(out_dir / "feature_importance_lgbm.csv", index=False)

    oof_df["y_true"] = y.values
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "comparator_cv_results.csv", index=False)

    summary = results_df.groupby("model")[["auroc", "average_precision", "brier_score"]].agg(["mean", "std"])
    print("\n[RESULTS]")
    print(summary.to_string())
    print(f"\n[INFO] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
