#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
supervised_backmapping.py

Trains supervised classifiers to recapitulate embedding-derived phenotype labels
from structured clinical variables. This serves as an interpretability audit of
embedding-derived phenotypes — it characterizes which structured variables are
most informative for discriminating the embedding-derived phenotype groups.

This is not a prognostic prediction model and is not used to predict clinical
outcomes or mortality.

Outputs (in --out-dir):
  - backmapping_cv_results.csv      (fold, macro_auroc, macro_recall)
  - oof_predictions.csv             (admission_ID, true_cluster, predicted_cluster,
                                     prob_cluster_{k} per class)
  - feature_importance_lgbm.csv     (feature, mean_gain)
  - class_level_performance.csv     (cluster, auroc, recall)

Usage:
  python supervised_backmapping.py \\
    --feature-csv ./data/admission_level_base.csv \\
    --cluster-labels ./clustering_out/hdbscan_labels.csv \\
    --out-dir ./backmapping_out \\
    --cv-folds 5 \\
    --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

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
    p = argparse.ArgumentParser(
        description=(
            "Interpretability audit: trains LightGBM to recapitulate embedding-derived "
            "phenotype labels from structured variables. "
            "This is NOT a prognostic prediction model."
        )
    )
    p.add_argument("--feature-csv", type=str, required=True,
                   help="admission_level_base.csv with structured clinical features")
    p.add_argument("--cluster-labels", type=str, required=True,
                   help="hdbscan_labels.csv from clustering step (embedding-derived labels)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write all outputs")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of stratified k-fold CV folds (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def load_data(feature_csv: Path, cluster_csv: Path):
    """Load structured features and embedding-derived cluster labels.

    Returns
    -------
    X : np.ndarray
        Preprocessed feature matrix.
    y : np.ndarray
        Embedding-derived cluster labels (integer).
    admission_ids : np.ndarray
        admission_ID values aligned with X and y.
    feature_names_used : list of str
        Feature column names corresponding to X columns.
    """
    feat_df = pd.read_csv(feature_csv, low_memory=False)
    feat_df["admission_ID"] = pd.to_numeric(feat_df["admission_ID"], errors="coerce")

    cluster_df = pd.read_csv(cluster_csv)
    cluster_df["admission_ID"] = pd.to_numeric(cluster_df["admission_ID"], errors="coerce")

    df = feat_df.merge(cluster_df[["admission_ID", "cluster_label"]], on="admission_ID", how="inner")

    # Exclude noise points (HDBSCAN label = -1)
    df = df[df["cluster_label"] != -1].reset_index(drop=True)
    print(f"[INFO] Merged: n={len(df)}, unique clusters={df['cluster_label'].nunique()}")

    available = [c for c in STRUCTURED_FEATURES if c in df.columns]
    missing = set(STRUCTURED_FEATURES) - set(available)
    if missing:
        print(f"[WARN] Missing feature columns (skipped): {sorted(missing)}")

    X_raw = df[available].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_raw))

    y = df["cluster_label"].values.astype(int)
    admission_ids = df["admission_ID"].values

    return X, y, admission_ids, available


def evaluate_fold_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray, classes: np.ndarray) -> dict:
    """Compute macro AUROC and macro recall for a single fold.

    Parameters
    ----------
    y_true : array of true cluster labels
    y_pred : array of predicted cluster labels
    y_prob : array of shape (n_samples, n_classes) with predicted probabilities
    classes : array of class labels

    Returns
    -------
    dict with macro_auroc, macro_recall, per_class_auroc (dict)
    """
    y_bin = label_binarize(y_true, classes=classes)

    if len(classes) == 2:
        # roc_auc_score expects 1D for binary
        macro_auroc = roc_auc_score(y_true, y_prob[:, 1])
        per_class_auroc = {
            int(classes[0]): roc_auc_score((y_true == classes[0]).astype(int), y_prob[:, 0]),
            int(classes[1]): roc_auc_score((y_true == classes[1]).astype(int), y_prob[:, 1]),
        }
    else:
        try:
            macro_auroc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
            per_class_auroc = {}
            for i, cl in enumerate(classes):
                try:
                    per_class_auroc[int(cl)] = roc_auc_score(y_bin[:, i], y_prob[:, i])
                except Exception:
                    per_class_auroc[int(cl)] = np.nan
        except Exception as e:
            print(f"[WARN] AUROC computation failed: {e}")
            macro_auroc = np.nan
            per_class_auroc = {int(cl): np.nan for cl in classes}

    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "macro_auroc": macro_auroc,
        "macro_recall": macro_recall,
        "per_class_auroc": per_class_auroc,
    }


def run_lgbm_cv(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, seed: int):
    """Run LightGBM with stratified k-fold cross-validation.

    Returns
    -------
    fold_results : list of dicts (one per fold)
    oof_predictions : dict with arrays for true labels, predicted labels, probabilities
    fi_df : DataFrame with mean feature importance (gain) across folds
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed. Run: pip install lightgbm", file=sys.stderr)
        sys.exit(1)

    classes = np.unique(y)
    n_classes = len(classes)
    n_samples = len(y)

    oof_true = np.full(n_samples, -1, dtype=int)
    oof_pred = np.full(n_samples, -1, dtype=int)
    oof_prob = np.full((n_samples, n_classes), np.nan)

    fold_results = []
    importances = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if n_classes == 2:
            objective = "binary"
        else:
            objective = "multiclass"

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            objective=objective,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        proba = model.predict_proba(X_te)
        pred = model.predict(X_te)

        oof_true[te_idx] = y_te
        oof_pred[te_idx] = pred.astype(int)
        oof_prob[te_idx] = proba

        fold_metrics = evaluate_fold_multiclass(y_te, pred, proba, classes)
        fold_results.append({
            "fold": fold,
            "macro_auroc": fold_metrics["macro_auroc"],
            "macro_recall": fold_metrics["macro_recall"],
        })
        print(f"  [LGBM] fold={fold}  macro_AUROC={fold_metrics['macro_auroc']:.3f}  "
              f"macro_recall={fold_metrics['macro_recall']:.3f}")

        importances.append(pd.Series(model.feature_importances_, name=f"fold_{fold}"))

    fi_df = pd.concat(importances, axis=1)
    fi_df.index = list(range(X.shape[1]))
    fi_mean = fi_df.mean(axis=1)

    oof_predictions = {
        "oof_true": oof_true,
        "oof_pred": oof_pred,
        "oof_prob": oof_prob,
        "classes": classes,
    }

    return fold_results, oof_predictions, fi_mean


def main():
    # This is an interpretability audit.
    # Target variable: embedding-derived cluster label (not mortality or clinical outcome).
    # Do not use these outputs for prognostic prediction or clinical decision support.

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    X, y, admission_ids, feature_names = load_data(
        Path(args.feature_csv), Path(args.cluster_labels)
    )
    print(f"[INFO] Features: {len(feature_names)}, samples: {len(y)}, classes: {np.unique(y)}")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    print(f"\n[INFO] Running LightGBM {args.cv_folds}-fold CV (interpretability audit)...")
    print("[INFO] Target: embedding-derived phenotype cluster label")
    fold_results, oof_preds, fi_mean = run_lgbm_cv(X, y, cv, args.seed)

    # Save CV results
    cv_df = pd.DataFrame(fold_results)
    cv_df.to_csv(out_dir / "backmapping_cv_results.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'backmapping_cv_results.csv'}")

    # Save OOF predictions
    classes = oof_preds["classes"]
    oof_df = pd.DataFrame({
        "admission_ID": admission_ids,
        "true_cluster": oof_preds["oof_true"],
        "predicted_cluster": oof_preds["oof_pred"],
    })
    for i, cl in enumerate(classes):
        oof_df[f"prob_cluster_{cl}"] = oof_preds["oof_prob"][:, i]
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'oof_predictions.csv'}")

    # Save feature importance
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "mean_gain": fi_mean.values,
    }).sort_values("mean_gain", ascending=False)
    fi_df.to_csv(out_dir / "feature_importance_lgbm.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'feature_importance_lgbm.csv'}")

    # Class-level performance from OOF
    class_perf_records = []
    y_bin = label_binarize(oof_preds["oof_true"], classes=classes)
    for i, cl in enumerate(classes):
        try:
            auroc = roc_auc_score(y_bin[:, i], oof_preds["oof_prob"][:, i])
        except Exception:
            auroc = np.nan
        mask_cl = oof_preds["oof_true"] == cl
        recall = recall_score(
            (oof_preds["oof_true"] == cl).astype(int),
            (oof_preds["oof_pred"] == cl).astype(int),
            zero_division=0,
        )
        class_perf_records.append({
            "cluster": int(cl),
            "auroc": auroc,
            "recall": recall,
            "n": int(mask_cl.sum()),
        })
    class_perf_df = pd.DataFrame(class_perf_records)
    class_perf_df.to_csv(out_dir / "class_level_performance.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'class_level_performance.csv'}")

    print("\n[RESULTS] OOF summary across folds:")
    print(cv_df[["fold", "macro_auroc", "macro_recall"]].to_string(index=False))
    print(f"\n[INFO] Done. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
