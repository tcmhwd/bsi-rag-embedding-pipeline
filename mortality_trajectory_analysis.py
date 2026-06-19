#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mortality_trajectory_analysis.py

Descriptive in-hospital mortality trajectory analysis after phenotype assignment.

Focus:
  - In-hospital mortality with discharge alive treated as a competing event.
  - Early in-hospital mortality (0–30 days from index BSI).
  - Late in-hospital mortality (beyond 30 days from index BSI).
  - Descriptive trajectory summaries by phenotype cluster.
  - Gray test for comparing cumulative incidence curves across clusters.
  - Cause-specific Cox model (alive discharge censored at discharge).

This script performs descriptive analyses after phenotype assignment.
It is not used to optimize clustering parameters or to train a mortality
prediction model.

Note: Column names (--time-col, --event-col, --competing-col) are placeholders.
Users must adapt these to their local deidentified dataset column names.

Outputs (in --out-dir):
  - cumulative_incidence.csv
  - gray_test_results.csv
  - cause_specific_cox_results.csv  (if lifelines/statsmodels available)
  - early_mortality_summary.csv     (0 to --landmark-day)
  - late_mortality_summary.csv      (beyond --landmark-day)
  - cluster_mortality_summary.csv

Usage:
  python mortality_trajectory_analysis.py \\
    --feature-csv ./data/admission_level_base.csv \\
    --cluster-labels ./clustering_out/hdbscan_labels.csv \\
    --out-dir ./trajectories_out \\
    --time-col days_from_index_bsi_to_death_or_discharge \\
    --event-col inhospital_mortality \\
    --competing-col discharge_alive \\
    --landmark-day 30 \\
    --reference-cluster 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Descriptive in-hospital mortality trajectory analysis by BSI phenotype cluster. "
            "This is not a mortality prediction model."
        )
    )
    p.add_argument("--feature-csv", type=str, required=True,
                   help="admission_level_base.csv with clinical variables and outcome fields")
    p.add_argument("--cluster-labels", type=str, required=True,
                   help="hdbscan_labels.csv with cluster_label column")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write all outputs")
    p.add_argument(
        "--time-col", type=str,
        default="days_from_index_bsi_to_death_or_discharge",
        help=(
            "Column name for time from index BSI to death or discharge in days. "
            "Default: 'days_from_index_bsi_to_death_or_discharge'. "
            "Adapt to your local deidentified dataset column name."
        ),
    )
    p.add_argument(
        "--event-col", type=str,
        default="inhospital_mortality",
        help="1=died in hospital, 0=survived/censored. Adapt to your local column name.",
    )
    p.add_argument(
        "--competing-col", type=str,
        default="discharge_alive",
        help="1=discharged alive (competing event), 0=otherwise. Adapt to your local column name.",
    )
    p.add_argument(
        "--landmark-day", type=int, default=30,
        help="Landmark day separating early from late in-hospital mortality (default: 30)",
    )
    p.add_argument(
        "--reference-cluster", type=int, default=0,
        help="Reference cluster for cause-specific Cox model (default: 0)",
    )
    return p.parse_args()


def load_data(feature_csv: Path, cluster_csv: Path,
              time_col: str, event_col: str, competing_col: str) -> pd.DataFrame:
    """Load and merge feature CSV with cluster labels.

    Returns merged DataFrame with cluster_label, time, event, competing columns.
    """
    feat_df = pd.read_csv(feature_csv, low_memory=False)
    feat_df["admission_ID"] = pd.to_numeric(feat_df["admission_ID"], errors="coerce")

    cluster_df = pd.read_csv(cluster_csv)
    cluster_df["admission_ID"] = pd.to_numeric(cluster_df["admission_ID"], errors="coerce")

    df = feat_df.merge(cluster_df[["admission_ID", "cluster_label"]], on="admission_ID", how="inner")
    df = df[df["cluster_label"] != -1].reset_index(drop=True)
    print(f"[INFO] Merged: n={len(df)}, clusters={df['cluster_label'].nunique()}")

    for col in [time_col, event_col]:
        if col not in df.columns:
            print(f"[ERROR] Required column '{col}' not found in merged data. "
                  f"Use --time-col / --event-col to specify the correct column name.",
                  file=sys.stderr)
            sys.exit(1)

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    if competing_col in df.columns:
        df[competing_col] = pd.to_numeric(df[competing_col], errors="coerce")
    else:
        print(f"[WARN] Competing event column '{competing_col}' not found. "
              f"Treating all non-events as censored. "
              f"Use --competing-col to specify the correct column name.")
        df[competing_col] = 0

    df = df.dropna(subset=[time_col, event_col]).reset_index(drop=True)
    return df


def compute_cumulative_incidence_by_cluster(df: pd.DataFrame, time_col: str,
                                             event_col: str,
                                             competing_col: str) -> pd.DataFrame:
    """Compute Aalen-Johansen cumulative incidence for each cluster.

    Uses lifelines.AalenJohansenFitter.
    Returns DataFrame with columns: cluster, time, cumulative_incidence.
    """
    try:
        from lifelines import AalenJohansenFitter
    except ImportError:
        print("[ERROR] lifelines not installed. Run: pip install lifelines", file=sys.stderr)
        sys.exit(1)

    # Encode: 0=censored, 1=primary event (in-hospital death), 2=competing (alive discharge)
    event_type = np.where(
        df[event_col] == 1, 1,
        np.where(df[competing_col] == 1, 2, 0)
    )
    T = df[time_col].clip(lower=0)

    records = []
    for cluster in sorted(df["cluster_label"].unique()):
        mask = df["cluster_label"] == cluster
        aj = AalenJohansenFitter(calculate_variance=True)
        aj.fit(T[mask], event_type[mask], event_of_interest=1, label=f"cluster_{cluster}")
        ci = aj.cumulative_density_.reset_index()
        ci.columns = ["time", "cumulative_incidence"]
        ci["cluster"] = cluster
        records.append(ci)

    return pd.concat(records, ignore_index=True)


def run_gray_test(df: pd.DataFrame, time_col: str, event_col: str,
                  competing_col: str) -> pd.DataFrame:
    """Run Gray test comparing cumulative incidence curves across clusters.

    Uses the R-equivalent approach via lifelines statistics if available.
    Returns DataFrame with Gray test results.
    """
    try:
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        print("[WARN] lifelines not available; skipping Gray test.", file=sys.stderr)
        return pd.DataFrame({"note": ["lifelines not installed; Gray test not run"]})

    event_type = np.where(
        df[event_col] == 1, 1,
        np.where(df[competing_col] == 1, 2, 0)
    )
    T = df[time_col].clip(lower=0)
    groups = df["cluster_label"].values

    try:
        result = multivariate_logrank_test(
            T, groups,
            event_observed=(event_type == 1).astype(int),
        )
        gray_df = pd.DataFrame([{
            "test": "Gray_test_competing_risk",
            "test_statistic": result.test_statistic,
            "p_value": result.p_value,
            "n_groups": df["cluster_label"].nunique(),
            "note": (
                "Approximate Gray test via multivariate log-rank on primary event indicator. "
                "For formal Gray test, use cmprsk R package or equivalent."
            ),
        }])
    except Exception as e:
        print(f"[WARN] Gray test failed: {e}")
        gray_df = pd.DataFrame([{"test": "Gray_test_competing_risk", "error": str(e)}])

    return gray_df


def compute_cause_specific_cox(df: pd.DataFrame, time_col: str, event_col: str,
                                cluster_col: str,
                                reference_cluster: int) -> pd.DataFrame:
    """Cause-specific Cox model with cluster as predictor.

    Alive discharge events are censored (treated as censoring for the death hazard).
    Uses lifelines CoxPHFitter.

    Returns DataFrame with Cox model coefficients and confidence intervals.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("[WARN] lifelines not installed; skipping cause-specific Cox model.", file=sys.stderr)
        return pd.DataFrame({"note": ["lifelines not installed; Cox model not run"]})

    df_cox = df[[time_col, event_col, cluster_col]].copy()
    df_cox = df_cox[df_cox[cluster_col] != reference_cluster].copy()

    # One-hot encode clusters (relative to reference)
    clusters = sorted(df[cluster_col].unique())
    non_ref = [c for c in clusters if c != reference_cluster]

    df_full = df[[time_col, event_col, cluster_col]].copy()
    for cl in non_ref:
        df_full[f"cluster_{cl}_vs_{reference_cluster}"] = (df_full[cluster_col] == cl).astype(int)

    covariate_cols = [f"cluster_{cl}_vs_{reference_cluster}" for cl in non_ref]
    df_model = df_full[[time_col, event_col] + covariate_cols].dropna()

    try:
        cph = CoxPHFitter()
        cph.fit(df_model, duration_col=time_col, event_col=event_col)
        cox_df = cph.summary.reset_index()
        cox_df.rename(columns={"index": "covariate"}, inplace=True)
        cox_df["reference_cluster"] = reference_cluster
    except Exception as e:
        print(f"[WARN] Cox model failed: {e}")
        cox_df = pd.DataFrame([{"error": str(e)}])

    return cox_df


def split_early_late(df: pd.DataFrame, time_col: str, event_col: str,
                     landmark_day: int):
    """Split events into early (0 to landmark_day) and late (beyond landmark_day).

    Returns (early_df, late_df).
    """
    # Early: event occurred on or before landmark_day
    early_df = df.copy()
    early_df[event_col] = np.where(
        (df[event_col] == 1) & (df[time_col] <= landmark_day), 1, 0
    )
    early_df[time_col] = df[time_col].clip(upper=landmark_day)

    # Late: only those who survived past landmark_day; event occurs after
    survived_landmark = df[df[time_col] > landmark_day].copy()
    late_df = survived_landmark.copy()
    late_df[event_col] = np.where(
        (survived_landmark[event_col] == 1) & (survived_landmark[time_col] > landmark_day), 1, 0
    )
    late_df[time_col] = survived_landmark[time_col] - landmark_day

    return early_df, late_df


def cluster_mortality_summary(df: pd.DataFrame, time_col: str, event_col: str,
                               competing_col: str, landmark_day: int) -> pd.DataFrame:
    """Descriptive mortality summary by cluster.

    Returns DataFrame with per-cluster counts, overall mortality rate,
    early and late mortality rates, and median survival time.
    """
    early_df, late_df = split_early_late(df, time_col, event_col, landmark_day)

    rows = []
    for cluster in sorted(df["cluster_label"].unique()):
        mask = df["cluster_label"] == cluster
        early_mask = early_df["cluster_label"] == cluster
        late_mask = late_df["cluster_label"] == cluster

        n_total = int(mask.sum())
        n_events = int(df.loc[mask, event_col].sum())
        n_competing = int(df.loc[mask, competing_col].sum()) if competing_col in df.columns else np.nan
        n_early_events = int(early_df.loc[early_mask, event_col].sum()) if early_mask.any() else 0
        n_late_eligible = int(late_mask.sum())
        n_late_events = int(late_df.loc[late_mask, event_col].sum()) if late_mask.any() else 0

        rows.append({
            "cluster": cluster,
            "n": n_total,
            "n_inhospital_deaths": n_events,
            "n_discharge_alive": n_competing,
            "overall_mortality_rate": n_events / n_total if n_total > 0 else np.nan,
            f"early_mortality_rate_0_to_{landmark_day}d": n_early_events / n_total if n_total > 0 else np.nan,
            f"late_mortality_eligible_n": n_late_eligible,
            f"late_mortality_events_beyond_{landmark_day}d": n_late_events,
            f"late_mortality_rate_beyond_{landmark_day}d": (
                n_late_events / n_late_eligible if n_late_eligible > 0 else np.nan
            ),
            "median_time_days": df.loc[mask, time_col].median(),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    df = load_data(
        Path(args.feature_csv), Path(args.cluster_labels),
        args.time_col, args.event_col, args.competing_col,
    )

    print(f"[INFO] n={len(df)}, clusters={df['cluster_label'].nunique()}, "
          f"event_rate={df[args.event_col].mean():.3f}")

    # Cumulative incidence
    print("[INFO] Computing cumulative incidence by cluster...")
    ci_df = compute_cumulative_incidence_by_cluster(
        df, args.time_col, args.event_col, args.competing_col
    )
    ci_df.to_csv(out_dir / "cumulative_incidence.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'cumulative_incidence.csv'}")

    # Gray test
    print("[INFO] Running Gray test...")
    gray_df = run_gray_test(df, args.time_col, args.event_col, args.competing_col)
    gray_df.to_csv(out_dir / "gray_test_results.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'gray_test_results.csv'}")

    # Cause-specific Cox
    print("[INFO] Fitting cause-specific Cox model...")
    cox_df = compute_cause_specific_cox(
        df, args.time_col, args.event_col, "cluster_label", args.reference_cluster
    )
    cox_df.to_csv(out_dir / "cause_specific_cox_results.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'cause_specific_cox_results.csv'}")

    # Early and late mortality
    print(f"[INFO] Splitting early/late mortality at landmark day={args.landmark_day}...")
    early_df, late_df = split_early_late(df, args.time_col, args.event_col, args.landmark_day)

    def _early_summary(grp_df, cluster_col="cluster_label"):
        rows = []
        for cl in sorted(grp_df[cluster_col].unique()):
            sub = grp_df[grp_df[cluster_col] == cl]
            rows.append({
                "cluster": cl,
                "n": len(sub),
                "n_events": int(sub[args.event_col].sum()),
                "mortality_rate": sub[args.event_col].mean(),
                "median_time": sub[args.time_col].median(),
            })
        return pd.DataFrame(rows)

    early_sum = _early_summary(early_df)
    early_sum.to_csv(out_dir / "early_mortality_summary.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'early_mortality_summary.csv'}")

    late_sum = _early_summary(late_df)
    late_sum.to_csv(out_dir / "late_mortality_summary.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'late_mortality_summary.csv'}")

    # Cluster mortality summary
    print("[INFO] Generating cluster mortality summary...")
    summary_df = cluster_mortality_summary(
        df, args.time_col, args.event_col, args.competing_col, args.landmark_day
    )
    summary_df.to_csv(out_dir / "cluster_mortality_summary.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'cluster_mortality_summary.csv'}")

    print(f"\n[INFO] Done. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
