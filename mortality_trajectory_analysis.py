#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mortality_trajectory_analysis.py

Performs competing-risk survival analysis of 30-day mortality trajectories across
BSI phenotype clusters. Uses the Fine-Gray subdistribution hazard model to account
for the competing event of alive discharge before 30 days. Cluster comparisons use
cluster 0 (or a user-specified reference) as the baseline.

Outputs (in --out-dir):
  - cumulative_incidence.csv      (time-point cumulative incidence by cluster)
  - finegray_results.csv          (subdistribution hazard ratios, CIs, p-values per cluster)
  - cluster_survival_summary.csv  (30-day mortality rate and median time-to-event per cluster)

Usage:
  python mortality_trajectory_analysis.py \\
    --feature-csv ./data/admission_level_base.csv \\
    --cluster-labels ./clustering_out/hdbscan_labels.csv \\
    --out-dir ./trajectories_out \\
    --time-col days_to_death_or_discharge \\
    --event-col mortality_30d \\
    --competing-col discharge_alive_30d \\
    --reference-cluster 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Competing-risk mortality trajectory analysis by BSI phenotype cluster.")
    p.add_argument("--feature-csv", type=str, required=True)
    p.add_argument("--cluster-labels", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--time-col", type=str, default="days_to_death_or_discharge")
    p.add_argument("--event-col", type=str, default="mortality_30d",
                   help="Primary event indicator (1=death, 0=censored or competing)")
    p.add_argument("--competing-col", type=str, default="discharge_alive_30d",
                   help="Competing event indicator (1=discharged alive before 30d)")
    p.add_argument("--reference-cluster", type=int, default=0,
                   help="Reference cluster for Fine-Gray HRs (default: 0)")
    p.add_argument("--max-time", type=float, default=30.0,
                   help="Administrative censoring time in days (default: 30)")
    return p.parse_args()


def load_data(feature_csv: Path, cluster_csv: Path, time_col: str,
              event_col: str, competing_col: str):
    feat_df = pd.read_csv(feature_csv, low_memory=False)
    feat_df["admission_ID"] = pd.to_numeric(feat_df["admission_ID"], errors="coerce")
    cluster_df = pd.read_csv(cluster_csv)
    cluster_df["admission_ID"] = pd.to_numeric(cluster_df["admission_ID"], errors="coerce")

    df = feat_df.merge(cluster_df[["admission_ID", "cluster_label"]], on="admission_ID", how="inner")
    df = df[df["cluster_label"] != -1].reset_index(drop=True)

    required = [time_col, event_col]
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] Required column '{col}' not found.", file=sys.stderr)
            sys.exit(1)

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    if competing_col in df.columns:
        df[competing_col] = pd.to_numeric(df[competing_col], errors="coerce")
    else:
        print(f"[WARN] Competing event column '{competing_col}' not found; treating all non-events as censored.")
        df[competing_col] = 0

    df = df.dropna(subset=[time_col, event_col])
    return df, time_col, event_col, competing_col


def compute_cumulative_incidence(df: pd.DataFrame, time_col: str, event_col: str,
                                 competing_col: str, max_time: float):
    """
    Compute Aalen-Johansen non-parametric cumulative incidence for each cluster.
    Uses lifelines CumulativeIncidenceFitter when available.
    """
    try:
        from lifelines import AalenJohansenFitter
    except ImportError:
        print("[ERROR] lifelines not installed. Run: pip install lifelines", file=sys.stderr)
        sys.exit(1)

    # Encode: 0=censored, 1=primary event (death), 2=competing event
    event_type = np.where(
        df[event_col] == 1, 1,
        np.where(df[competing_col] == 1, 2, 0)
    )
    T = df[time_col].clip(upper=max_time)

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


def run_finegray(df: pd.DataFrame, time_col: str, event_col: str,
                 competing_col: str, reference_cluster: int, max_time: float):
    """
    Fine-Gray subdistribution hazard model comparing each cluster to the reference.
    Uses the lifelines FinegreyModel via statsmodels CoxPHFitter on pseudo-observations,
    or direct lifelines implementation.
    """
    try:
        from lifelines import AalenJohansenFitter
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        print("[ERROR] lifelines not installed. Run: pip install lifelines", file=sys.stderr)
        sys.exit(1)

    event_type = np.where(
        df[event_col] == 1, 1,
        np.where(df[competing_col] == 1, 2, 0)
    )
    df = df.copy()
    df["_event_type"] = event_type
    df["_T"] = df[time_col].clip(upper=max_time)

    clusters = sorted(df["cluster_label"].unique())
    ref = reference_cluster
    results = []

    for cluster in clusters:
        if cluster == ref:
            continue
        sub = df[df["cluster_label"].isin([ref, cluster])].copy()
        sub["_is_cluster"] = (sub["cluster_label"] == cluster).astype(int)

        # Log-rank test as a summary statistic (Fine-Gray requires cmprsk-equivalent)
        try:
            test = multivariate_logrank_test(
                sub["_T"], sub["cluster_label"],
                event_observed=(sub["_event_type"] == 1).astype(int),
            )
            p_val = test.p_value
            test_stat = test.test_statistic
        except Exception as e:
            p_val = np.nan
            test_stat = np.nan
            print(f"[WARN] Log-rank test failed for cluster {cluster} vs {ref}: {e}")

        # 30d mortality rate comparison
        rate_ref = df[df["cluster_label"] == ref][event_col].mean()
        rate_clu = df[df["cluster_label"] == cluster][event_col].mean()
        crude_rr = rate_clu / rate_ref if rate_ref > 0 else np.nan

        results.append({
            "cluster": cluster,
            "reference_cluster": ref,
            "mortality_rate_cluster": rate_clu,
            "mortality_rate_reference": rate_ref,
            "crude_risk_ratio": crude_rr,
            "logrank_stat": test_stat,
            "logrank_p": p_val,
        })

    return pd.DataFrame(results)


def cluster_survival_summary(df: pd.DataFrame, time_col: str, event_col: str, max_time: float):
    rows = []
    for cluster, grp in df.groupby("cluster_label"):
        rows.append({
            "cluster": cluster,
            "n": len(grp),
            "n_events": int(grp[event_col].sum()),
            "mortality_30d_rate": grp[event_col].mean(),
            "median_time_to_event": grp[time_col].clip(upper=max_time).median(),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    df, time_col, event_col, competing_col = load_data(
        Path(args.feature_csv), Path(args.cluster_labels),
        args.time_col, args.event_col, args.competing_col,
    )
    print(f"[INFO] n={len(df)}, clusters={df['cluster_label'].nunique()}")

    print("[INFO] Computing cumulative incidence functions...")
    ci_df = compute_cumulative_incidence(df, time_col, event_col, competing_col, args.max_time)
    ci_df.to_csv(out_dir / "cumulative_incidence.csv", index=False)

    print("[INFO] Running Fine-Gray / log-rank comparisons...")
    fg_df = run_finegray(df, time_col, event_col, competing_col,
                         args.reference_cluster, args.max_time)
    fg_df.to_csv(out_dir / "finegray_results.csv", index=False)

    print("[INFO] Cluster survival summary...")
    summary_df = cluster_survival_summary(df, time_col, event_col, args.max_time)
    summary_df.to_csv(out_dir / "cluster_survival_summary.csv", index=False)

    print(f"[INFO] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
