#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_case_reports.py (de-ID + leakage-reduced, updated)

Key updates (2026-02)
---------------------
- Physiology/labs reduced to a minimal set:
  keep: SBP, RR, PR, Body_temperature, GCS_2, WBC_Count, CRP, Creatinine,
        T_Bilirubin, PLT_Count, Albumin, Hemoglobin
  drop: MBP, FiO2, pao2/fio2
  optional: AST/ALT (enable with --include-transaminases)

- Administrative section: remove "Entry mode"
- Organ support section: omit ventilator/line/catheter if NA (keep only informative yes/no)

Inputs
------
--data-dir must contain:
  - admission_level_base.csv   (admission-level feature table)
  - bsi_with_ast_tt.csv        (episode-level microbiology / AST table)

Outputs
-------
- {out_dir}/{newpatient_ID}_{admission_ID}.report.md

Usage
-----
Single:
  python make_case_reports.py \
    --data-dir ./data \
    --admission-id 12345 \
    --out-dir ./reports_single

All:
  python make_case_reports.py \
    --data-dir ./data \
    --all \
    --out-dir ./reports_all
"""

import argparse
from pathlib import Path
import re
import pandas as pd

AST_WHITELIST = [
    "Ampicillin",
    "Meropenem",
    "Ampicillin Sulbactam",
    "Sulfamethoxazole trimethoprim",
    "Cefotaxime",
    "Levofloxacin",
    "Piperacillin Tazobactam",
    "Ceftazidime",
    "Ertapenem",
    "Ciprofloxacin",
    "Penicillin G",
    "Tetracycline",
    "Vancomycin",
    "Oxacillin",
    "Ceftriaxone",
    "Amoxicillin clavulanate",
]

AGE_BINS = [(18, 39, "18–39"), (40, 64, "40–64"), (65, 74, "65–74"), (75, 84, "75–84")]

# Minimal lab set (default)
MIN_LAB_VARS = [
    "SBP",
    "RR",
    "PR",
    "Body_temperature",
    "GCS_2",
    "WBC_Count",
    "CRP",
    "Creatinine",
    "T_Bilirubin",
    "PLT_Count",
    "Albumin",
    "Hemoglobin",
]


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate de-identified BSI-only case report(s).")
    p.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing admission_level_base.csv and bsi_with_ast_tt.csv",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--admission-id", type=int, help="Generate report for a single admission_ID")
    g.add_argument("--all", action="store_true", help="Generate reports for all admissions")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p.add_argument("--max-n", type=int, default=None, help="If --all, generate only first N admissions (debug)")
    p.add_argument(
        "--include-transaminases",
        action="store_true",
        help="Include AST/ALT in Physiology and labs (default: off).",
    )
    return p.parse_args()


# -------------------------
# Utils
# -------------------------
def val_or_na(v):
    if pd.isna(v):
        return "NA"
    if isinstance(v, str) and v.strip() == "":
        return "NA"
    return str(v)


def yesno_from_binary(v):
    if pd.isna(v):
        return "NA"
    try:
        iv = int(v)
        return "yes" if iv == 1 else "no"
    except Exception:
        s = str(v).strip().lower()
        if s in ["yes", "y", "true", "1"]:
            return "yes"
        if s in ["no", "n", "false", "0"]:
            return "no"
        return "NA"


def age_to_band(age_val):
    if pd.isna(age_val):
        return "NA"
    try:
        age = float(age_val)
    except Exception:
        return "NA"
    if age < 18:
        return "<18"
    for lo, hi, lab in AGE_BINS:
        if lo <= age <= hi:
            return lab
    return "≥85"


def admission_year(admission_date):
    if pd.isna(admission_date):
        return "NA"
    d = pd.to_datetime(admission_date, errors="coerce")
    if pd.isna(d):
        return "NA"
    return str(int(d.year))


def hospital_day(test_date, admission_date):
    """
    Return "hospital day X" where X is (test_date - admission_date).days,
    but if negative, cap to 0.
    """
    if pd.isna(test_date) or pd.isna(admission_date):
        return "NA"
    t = pd.to_datetime(test_date, errors="coerce")
    a = pd.to_datetime(admission_date, errors="coerce")
    if pd.isna(t) or pd.isna(a):
        return "NA"
    delta = (t.normalize() - a.normalize()).days
    if delta < 0:
        delta = 0
    return f"hospital day {delta}"


def parse_ab_list(raw):
    if not isinstance(raw, str):
        return []
    xs = [x.strip() for x in raw.split(",") if x.strip()]
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def include_yesno_if_informative(name: str, raw_val) -> str | None:
    """
    For device/line variables where NA is common:
    - include only if value can be interpreted as yes/no (not NA)
    """
    yn = yesno_from_binary(raw_val)
    if yn == "NA":
        return None
    return f"{name}: {yn}"


# -------------------------
# Load tables
# -------------------------
def load_all_tables(data_dir: Path):
    adm_path = data_dir / "admission_level_base.csv"
    bsi_path = data_dir / "bsi_with_ast_tt.csv"

    print(f"[INFO] Loading admission_level_base: {adm_path}")
    df_adm = pd.read_csv(adm_path, low_memory=False)

    print(f"[INFO] Loading bsi_with_ast_tt (episode-level): {bsi_path}")
    df_bsi = pd.read_csv(bsi_path, low_memory=False)

    for df, cols in [(df_adm, ["admission_date", "test_date"]), (df_bsi, ["admission_date", "test_date"])]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    df_adm["admission_ID"] = pd.to_numeric(df_adm["admission_ID"], errors="coerce")
    df_bsi["admission_ID"] = pd.to_numeric(df_bsi["admission_ID"], errors="coerce")

    return df_adm, df_bsi


# -------------------------
# AST helpers
# -------------------------
def parse_ast_by_microbe(ast_by_microbe: str):
    """
    Format assumed:
      microbe_x{Ab1:R; Ab2:S} | microbe_y{NA} | ...
    """
    if not isinstance(ast_by_microbe, str) or ast_by_microbe.strip() == "":
        return {}
    out = {}
    blocks = [b.strip() for b in ast_by_microbe.split("|") if b.strip()]
    for blk in blocks:
        m = re.match(r"^\s*([^{}]+)\{(.*)\}\s*$", blk)
        if not m:
            continue
        microbe = m.group(1).strip()
        content = m.group(2).strip()
        if content.upper() == "NA" or content == "":
            out[microbe] = []
            continue
        out[microbe] = [p.strip() for p in content.split(";") if p.strip()]
    return out


def filter_ast_pairs(pairs, whitelist):
    keep = []
    for p in pairs:
        if ":" not in p:
            continue
        ab, res = p.split(":", 1)
        ab = ab.strip()
        res = res.strip()
        if ab in whitelist:
            keep.append((ab, res))
    return keep


def parse_ast_episode_flat(ast_episode: str):
    """
    Format assumed:
      Ab1:R; Ab2:S; ...
    """
    if not isinstance(ast_episode, str) or ast_episode.strip() == "":
        return []
    parts = [x.strip() for x in ast_episode.split(";") if x.strip()]
    pairs = []
    for p in parts:
        if ":" not in p:
            continue
        ab, res = p.split(":", 1)
        pairs.append((ab.strip(), res.strip()))
    return pairs


# -------------------------
# Early empiric antibiotics
# -------------------------
def get_early_empiric_ab_names(epi_row: pd.Series):
    """
    Prefer empiric72 -> empiric48 -> empiric24.
    Fallback to tt_ab_name_list_episode only if empiric columns are absent.
    """
    for col in ["ab_empiric72_name_clean", "ab_empiric48_name_clean", "ab_empiric24_name_clean"]:
        if col in epi_row.index:
            names = parse_ab_list(epi_row.get(col))
            if names:
                return names

    # fallback
    for col in ["tt_ab_name_list_episode", "tt_ab_name_list"]:
        if col in epi_row.index:
            names = parse_ab_list(epi_row.get(col))
            if names:
                return names

    return []


# =========================
# Section builders
# =========================
def build_section_admin(adm_row: pd.Series) -> str:
    newpatient_id = val_or_na(adm_row.get("newpatient_ID"))
    age_band = age_to_band(adm_row.get("age_at_test"))
    sex = val_or_na(adm_row.get("sex"))
    adm_year = admission_year(adm_row.get("admission_date"))

    # Entry mode intentionally removed (noise / site-specific).
    dept = val_or_na(adm_row.get("treatment_department_eng"))

    lines = [
        "[Administrative information (de-identified)]",
        f"Patient pseudo-ID: {newpatient_id}",
        f"Age group: {age_band}",
        f"Sex: {sex}",
        f"Admission year: {adm_year}",
        f"Treatment department: {dept}",
    ]
    return "\n".join(lines)


def build_section_fungemia(adm_row: pd.Series) -> str:
    """
    Expecting fields like:
      - is_fungemia_admission (0/1) OR fungemia (0/1)
      - fungemia_organism
      - fungemia_date (hospital day integer)
    """
    # Try common column names
    f_flag = None
    for c in ["is_fungemia_admission", "fungemia", "fungemia_admission"]:
        if c in adm_row.index:
            f_flag = c
            break

    flag = yesno_from_binary(adm_row.get(f_flag)) if f_flag else "NA"
    org = val_or_na(adm_row.get("fungemia_organism")) if "fungemia_organism" in adm_row.index else "NA"
    fday = val_or_na(adm_row.get("fungemia_date")) if "fungemia_date" in adm_row.index else "NA"

    lines = ["[Fungemia (known at or before index culture)]"]
    lines.append(f"Fungemia: {flag}")
    if flag == "yes":
        # organism/date are useful only if positive
        lines.append(f"Fungemia organism: {org}")
        lines.append(f"Fungemia timing: hospital day {fday}" if fday != "NA" else "Fungemia timing: NA")
    return "\n".join(lines)


def build_section_comorbidities(adm_row: pd.Series) -> str:
    icd_inf = val_or_na(adm_row.get("icd_categorized_onlyinfections"))
    icd_all = val_or_na(adm_row.get("icd_all_codes"))
    charlson_cat = val_or_na(adm_row.get("charlson_category_withcancermerged", adm_row.get("charlson_category")))
    charlson_score = val_or_na(adm_row.get("charlson_score"))

    lines = [
        "[Comorbidities]",
        f"Infectious ICD categories: {icd_inf}",
        f"All ICD10 codes (during admission): {icd_all}",
        f"Charlson categories: {charlson_cat}",
        f"Charlson score: {charlson_score}",
    ]
    return "\n".join(lines)


def build_section_severity_support(adm_row: pd.Series) -> str:
    """
    Keep core severity/support signals.
    Drop noisy devices if NA (only keep yes/no).
    """
    lines = ["[Organ support and severity scores (at index time if available)]"]

    # Core: keep even if NA (they define context)
    lines.append(f"icu: {yesno_from_binary(adm_row.get('icu'))}")
    lines.append(f"Vasopressin_any: {yesno_from_binary(adm_row.get('Vasopressin_any'))}")

    # Devices: include only if informative (not NA)
    for col in ["ventilator", "central_orarterial_line", "urinary_orrenal_catheter"]:
        if col in adm_row.index:
            s = include_yesno_if_informative(col, adm_row.get(col))
            if s is not None:
                lines.append(s)

    # Scores
    for col in ["sofa_score_2", "sepsis_2", "quick_sofa_score_2", "sepsis_quicksofa_2"]:
        if col in adm_row.index:
            lines.append(f"{col}: {val_or_na(adm_row.get(col))}")
        else:
            lines.append(f"{col}: NA")

    return "\n".join(lines)


def build_section_physio_minimal(adm_row: pd.Series, include_transaminases: bool) -> str:
    vars_needed = MIN_LAB_VARS.copy()
    if include_transaminases:
        # Optional add-on (explicitly requested by CLI only)
        vars_needed += ["AST", "ALT"]

    lines = ["[Physiology and labs (available at index time if provided)]"]
    for v in vars_needed:
        if v in adm_row.index:
            lines.append(f"{v}: {val_or_na(adm_row.get(v))}")
        else:
            lines.append(f"{v}: NA")
    return "\n".join(lines)


def build_section_colonization_spectrum(adm_row: pd.Series) -> str:
    lines = ["[Colonization and resistance/spectrum flags]"]

    for col in ["VRE_colonization", "CRE_colonization"]:
        if col in adm_row.index:
            lines.append(f"{col}: {val_or_na(adm_row.get(col))}")

    # empiric spectrum flags
    for col in ["antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric"]:
        if col in adm_row.index:
            lines.append(f"{col}: {yesno_from_binary(adm_row.get(col))}")

    # broadness flags (keep empiric; admission-wide may be time-mixed but user asked to keep others)
    for col in ["is_empiric_tt_excessivelylarge", "is_admission_tt_excessivelylarge"]:
        if col in adm_row.index:
            lines.append(f"{col}: {yesno_from_binary(adm_row.get(col))}")

    return "\n".join(lines)


def build_section_bsi_episodes(admission_id: int, admission_date, df_bsi: pd.DataFrame) -> str:
    lines = ["[Bloodstream infection episodes (microbiology/AST)]"]

    sub = df_bsi[df_bsi["admission_ID"] == admission_id].copy()
    if sub.empty:
        lines.append("No BSI episodes recorded for this admission (NA)")
        return "\n".join(lines)

    sub["test_date"] = pd.to_datetime(sub["test_date"], errors="coerce")
    sort_cols = ["test_date"]
    if "BSI_episode_num" in sub.columns:
        sort_cols.append("BSI_episode_num")
    sub = sub.sort_values(sort_cols, na_position="last")

    # Print each episode in order
    episode_nums = (
        sorted(sub["BSI_episode_num"].dropna().unique().tolist())
        if "BSI_episode_num" in sub.columns
        else [1]
    )

    for ep in episode_nums:
        epi = sub[sub["BSI_episode_num"] == ep] if "BSI_episode_num" in sub.columns else sub
        if epi.empty:
            continue
        row = epi.iloc[0]

        def flag(col):
            return yesno_from_binary(row.get(col)) if col in epi.columns else "NA"

        hd = hospital_day(row.get("test_date"), admission_date)
        organisms = val_or_na(row.get("BSI_episode_short_clean", row.get("BSI_episode_short")))
        poly = flag("BSI_episode_polymicrobial")

        lines.append(f"Episode: {ep}")
        lines.append(f"- Timing: {hd}")
        lines.append(f"- Organism summary: {organisms}")
        lines.append(f"- Polymicrobial: {poly}")

        for col in [
            "BSI_episode_3GCR",
            "BSI_episode_CarbaR",
            "BSI_episode_MRSA",
            "BSI_episode_VRE",
            "BSI_episode_CeftriaxoneR",
            "BSI_episode_CRAB",
        ]:
            lines.append(f"- {col}: {flag(col)}")

        # Early empiric antibiotics
        ab_names = get_early_empiric_ab_names(row)
        lines.append("- Early empiric antibiotics (≤72h concept):")
        if not ab_names:
            lines.append("  o NA")
        else:
            for i, name in enumerate(ab_names, start=1):
                lines.append(f"  o Antibiotic_{i}: {name}")

        # AST
        lines.append("- AST (selected antibiotics):")
        ast_by_microbe = row.get("AST_ab_result_by_microbe", "")
        ast_episode = row.get("AST_ab_result_list_episode", "")

        wrote_any = False
        parsed = parse_ast_by_microbe(ast_by_microbe) if isinstance(ast_by_microbe, str) else {}
        if parsed:
            merged = []
            for mid in parsed:
                merged.extend(filter_ast_pairs(parsed[mid], AST_WHITELIST))
            merged = sorted(set(merged))
            if merged:
                wrote_any = True
                for ab, res in merged:
                    lines.append(f"  o {ab}: {res}")

        if not wrote_any:
            flat_pairs = parse_ast_episode_flat(ast_episode)
            flat_pairs = [(ab, res) for (ab, res) in flat_pairs if ab in AST_WHITELIST]
            flat_pairs = sorted(set(flat_pairs))
            if not flat_pairs:
                lines.append("  o NA")
            else:
                for ab, res in flat_pairs:
                    lines.append(f"  o {ab}: {res}")

        lines.append("")

    return "\n".join(lines).rstrip()


def build_section_additional_structured(adm_row: pd.Series) -> str:
    """
    Keep as-is (user asked to keep the rest).
    Prints a small set if present.
    """
    keys = []
    for col in [
        "ab_admission_category_clean",
        "ab_admission_name_clean",
    ]:
        if col in adm_row.index:
            v = val_or_na(adm_row.get(col))
            if v != "NA":
                keys.append((col, v))

    if not keys:
        return ""

    lines = ["[Additional structured metadata (filtered)]"]
    for k, v in keys:
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


# -------------------------
# Report builder
# -------------------------
def make_case_report_for_admission(admission_id: int, df_adm: pd.DataFrame, df_bsi: pd.DataFrame, include_transaminases: bool):
    adm_sub = df_adm[df_adm["admission_ID"] == admission_id]
    if adm_sub.empty:
        raise ValueError(f"admission_ID {admission_id} not found in admission_level_base.")
    adm_row = adm_sub.iloc[0]
    admission_date = adm_row.get("admission_date", pd.NaT)

    sections = [
        "[Case]",
        build_section_admin(adm_row),
        build_section_fungemia(adm_row),
        build_section_comorbidities(adm_row),
        build_section_severity_support(adm_row),
        build_section_physio_minimal(adm_row, include_transaminases=include_transaminases),
        build_section_colonization_spectrum(adm_row),
        build_section_bsi_episodes(admission_id, admission_date, df_bsi),
    ]

    extra = build_section_additional_structured(adm_row)
    if extra.strip():
        sections.append(extra)

    report_text = "\n".join([s for s in sections if s is not None and str(s).strip() != ""]).strip() + "\n"
    return report_text, adm_row


def save_report(report_text: str, adm_row: pd.Series, admission_id: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    newpatient_id = val_or_na(adm_row.get("newpatient_ID"))
    filename = f"{newpatient_id}_{admission_id}.report.md"
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return out_path


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)

    df_adm, df_bsi = load_all_tables(data_dir)

    if args.admission_id is not None:
        admission_id = int(args.admission_id)
        report_text, adm_row = make_case_report_for_admission(
            admission_id, df_adm, df_bsi, include_transaminases=args.include_transaminases
        )
        out_path = save_report(report_text, adm_row, admission_id, out_dir)
        print(f"[INFO] Saved: {out_path}")
        print("[INFO] Done.")
        return

    admissions = df_adm["admission_ID"].dropna().astype(int).unique().tolist()
    admissions = sorted(admissions)
    if args.max_n is not None:
        admissions = admissions[: int(args.max_n)]

    print(f"[INFO] Generating reports for {len(admissions)} admissions...")

    n_ok, n_fail = 0, 0
    for i, admission_id in enumerate(admissions, start=1):
        try:
            report_text, adm_row = make_case_report_for_admission(
                int(admission_id), df_adm, df_bsi, include_transaminases=args.include_transaminases
            )
            save_report(report_text, adm_row, int(admission_id), out_dir)
            n_ok += 1
            if i % 500 == 0:
                print(f"[INFO] Progress: {i}/{len(admissions)} (ok={n_ok}, fail={n_fail})")
        except Exception as e:
            n_fail += 1
            print(f"[WARN] Failed admission_ID={admission_id}: {e}")

    print(f"[INFO] Done. ok={n_ok}, fail={n_fail}, out_dir={out_dir}")


if __name__ == "__main__":
    main()

