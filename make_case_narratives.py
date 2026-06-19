#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_case_narratives.py

Converts structured hospitalization-level bloodstream infection (BSI) data into
standardized, outcome-stripped case narratives for embedding-based phenotyping.

Each narrative covers:
  - De-identified administrative information (age band, sex, admission year, department)
  - Comorbidities (Charlson categories and score, ICD-10 infection categories)
  - Severity and organ support (ICU, vasopressors, SOFA, qSOFA)
  - Physiology and labs at index culture time
  - Colonization / resistance flags
  - BSI episode(s): organism, resistance phenotype, early empiric antibiotics, AST

Outcome fields (mortality, discharge status, length of stay) are excluded
to prevent label leakage before embedding.

Inputs
------
--data-dir must contain:
  - admission_level_base.csv   (admission-level feature table)
  - bsi_with_ast_tt.csv        (episode-level microbiology / AST table)

Outputs
-------
  - {out_dir}/{newpatient_ID}_{admission_ID}.narrative.md

Usage
-----
Single:
  python make_case_narratives.py \\
    --data-dir ./data \\
    --admission-id 12345 \\
    --out-dir ./narratives

All:
  python make_case_narratives.py \\
    --data-dir ./data \\
    --all \\
    --out-dir ./narratives
"""

import argparse
import re
from pathlib import Path

import pandas as pd

AST_WHITELIST = [
    "Ampicillin", "Meropenem", "Ampicillin Sulbactam",
    "Sulfamethoxazole trimethoprim", "Cefotaxime", "Levofloxacin",
    "Piperacillin Tazobactam", "Ceftazidime", "Ertapenem",
    "Ciprofloxacin", "Penicillin G", "Tetracycline",
    "Vancomycin", "Oxacillin", "Ceftriaxone", "Amoxicillin clavulanate",
]

AGE_BINS = [(18, 39, "18–39"), (40, 64, "40–64"), (65, 74, "65–74"), (75, 84, "75–84")]

LAB_VARS = [
    "SBP", "RR", "PR", "Body_temperature", "GCS_2",
    "WBC_Count", "CRP", "Creatinine", "T_Bilirubin",
    "PLT_Count", "Albumin", "Hemoglobin",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate de-identified, outcome-stripped BSI case narrative(s)."
    )
    p.add_argument("--data-dir", type=str, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--admission-id", type=int)
    g.add_argument("--all", action="store_true")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--include-transaminases", action="store_true")
    return p.parse_args()


def val_or_na(v):
    if pd.isna(v):
        return "NA"
    if isinstance(v, str) and not v.strip():
        return "NA"
    return str(v)


def yesno(v):
    if pd.isna(v):
        return "NA"
    try:
        return "yes" if int(v) == 1 else "no"
    except Exception:
        s = str(v).strip().lower()
        return "yes" if s in ("yes", "y", "true", "1") else ("no" if s in ("no", "n", "false", "0") else "NA")


def age_band(age_val):
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


def adm_year(dt):
    d = pd.to_datetime(dt, errors="coerce")
    return "NA" if pd.isna(d) else str(int(d.year))


def hospital_day(test_dt, adm_dt):
    t, a = pd.to_datetime(test_dt, errors="coerce"), pd.to_datetime(adm_dt, errors="coerce")
    if pd.isna(t) or pd.isna(a):
        return "NA"
    return f"hospital day {max(0, (t.normalize() - a.normalize()).days)}"


def parse_ab_list(raw):
    if not isinstance(raw, str):
        return []
    seen, out = set(), []
    for x in (x.strip() for x in raw.split(",") if x.strip()):
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_ast_by_microbe(s):
    out = {}
    if not isinstance(s, str):
        return out
    for blk in (b.strip() for b in s.split("|") if b.strip()):
        m = re.match(r"^\s*([^{}]+)\{(.*)\}\s*$", blk)
        if not m:
            continue
        microbe, content = m.group(1).strip(), m.group(2).strip()
        out[microbe] = [] if content.upper() == "NA" else [p.strip() for p in content.split(";") if p.strip()]
    return out


def filter_ast(pairs, whitelist):
    return [(ab.strip(), res.strip()) for p in pairs if ":" in p
            for ab, res in [p.split(":", 1)] if ab.strip() in whitelist]


def load_tables(data_dir: Path):
    df_adm = pd.read_csv(data_dir / "admission_level_base.csv", low_memory=False)
    df_bsi = pd.read_csv(data_dir / "bsi_with_ast_tt.csv", low_memory=False)
    for df in (df_adm, df_bsi):
        for col in ("admission_date", "test_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    df_adm["admission_ID"] = pd.to_numeric(df_adm["admission_ID"], errors="coerce")
    df_bsi["admission_ID"] = pd.to_numeric(df_bsi["admission_ID"], errors="coerce")
    return df_adm, df_bsi


def build_narrative(admission_id: int, df_adm: pd.DataFrame, df_bsi: pd.DataFrame,
                    include_transaminases: bool = False):
    sub = df_adm[df_adm["admission_ID"] == admission_id]
    if sub.empty:
        raise ValueError(f"admission_ID {admission_id} not found")
    row = sub.iloc[0]
    adm_dt = row.get("admission_date", pd.NaT)

    sections = []

    # Administrative
    sections.append("\n".join([
        "[Administrative information (de-identified)]",
        f"Patient pseudo-ID: {val_or_na(row.get('newpatient_ID'))}",
        f"Age group: {age_band(row.get('age_at_test'))}",
        f"Sex: {val_or_na(row.get('sex'))}",
        f"Admission year: {adm_year(row.get('admission_date'))}",
        f"Treatment department: {val_or_na(row.get('treatment_department_eng'))}",
    ]))

    # Comorbidities
    sections.append("\n".join([
        "[Comorbidities]",
        f"Infectious ICD categories: {val_or_na(row.get('icd_categorized_onlyinfections'))}",
        f"All ICD10 codes: {val_or_na(row.get('icd_all_codes'))}",
        f"Charlson categories: {val_or_na(row.get('charlson_category_withcancermerged', row.get('charlson_category')))}",
        f"Charlson score: {val_or_na(row.get('charlson_score'))}",
    ]))

    # Severity / organ support
    sev = ["[Organ support and severity scores]",
           f"icu: {yesno(row.get('icu'))}",
           f"Vasopressin_any: {yesno(row.get('Vasopressin_any'))}"]
    for col in ("ventilator", "central_orarterial_line", "urinary_orrenal_catheter"):
        yn = yesno(row.get(col)) if col in row.index else "NA"
        if yn != "NA":
            sev.append(f"{col}: {yn}")
    for col in ("sofa_score_2", "sepsis_2", "quick_sofa_score_2", "sepsis_quicksofa_2"):
        sev.append(f"{col}: {val_or_na(row.get(col))}")
    sections.append("\n".join(sev))

    # Physiology / labs
    lab_vars = LAB_VARS + (["AST", "ALT"] if include_transaminases else [])
    lab_lines = ["[Physiology and labs (at index culture time)]"]
    for v in lab_vars:
        lab_lines.append(f"{v}: {val_or_na(row.get(v))}")
    sections.append("\n".join(lab_lines))

    # Colonization / resistance flags
    col_lines = ["[Colonization and resistance flags]"]
    for col in ("VRE_colonization", "CRE_colonization",
                "antiMRSAab_empiric", "antiVREab_empiric",
                "antipseudoab_empiric", "carbaab_empiric",
                "is_empiric_tt_excessivelylarge"):
        if col in row.index:
            col_lines.append(f"{col}: {val_or_na(row.get(col))}")
    sections.append("\n".join(col_lines))

    # BSI episodes
    epi_lines = ["[Bloodstream infection episodes]"]
    bsi_sub = df_bsi[df_bsi["admission_ID"] == admission_id].copy()
    if bsi_sub.empty:
        epi_lines.append("No BSI episodes recorded (NA)")
    else:
        bsi_sub = bsi_sub.sort_values(["test_date"] + (["BSI_episode_num"] if "BSI_episode_num" in bsi_sub.columns else []), na_position="last")
        eps = sorted(bsi_sub["BSI_episode_num"].dropna().unique()) if "BSI_episode_num" in bsi_sub.columns else [1]
        for ep in eps:
            erow = (bsi_sub[bsi_sub["BSI_episode_num"] == ep] if "BSI_episode_num" in bsi_sub.columns else bsi_sub).iloc[0]
            epi_lines += [
                f"Episode: {ep}",
                f"- Timing: {hospital_day(erow.get('test_date'), adm_dt)}",
                f"- Organism: {val_or_na(erow.get('BSI_episode_short_clean', erow.get('BSI_episode_short')))}",
                f"- Polymicrobial: {yesno(erow.get('BSI_episode_polymicrobial'))}",
            ]
            for flag_col in ("BSI_episode_3GCR", "BSI_episode_CarbaR", "BSI_episode_MRSA",
                             "BSI_episode_VRE", "BSI_episode_CeftriaxoneR", "BSI_episode_CRAB"):
                epi_lines.append(f"- {flag_col}: {yesno(erow.get(flag_col))}")

            ab_names = next(
                (parse_ab_list(erow.get(c)) for c in ("ab_empiric72_name_clean", "ab_empiric48_name_clean",
                                                       "ab_empiric24_name_clean", "tt_ab_name_list_episode")
                 if c in erow.index and parse_ab_list(erow.get(c))), []
            )
            epi_lines.append("- Early empiric antibiotics (≤72h):")
            for i, name in enumerate(ab_names or ["NA"], 1):
                epi_lines.append(f"  o Antibiotic_{i}: {name}")

            epi_lines.append("- AST (selected antibiotics):")
            parsed = parse_ast_by_microbe(erow.get("AST_ab_result_by_microbe", ""))
            ast_pairs = sorted(set(
                p for pairs in parsed.values() for p in filter_ast(pairs, AST_WHITELIST)
            )) if parsed else filter_ast(
                [x.strip() for x in str(erow.get("AST_ab_result_list_episode", "")).split(";") if x.strip()],
                AST_WHITELIST
            )
            if ast_pairs:
                for ab, res in ast_pairs:
                    epi_lines.append(f"  o {ab}: {res}")
            else:
                epi_lines.append("  o NA")
            epi_lines.append("")

    sections.append("\n".join(epi_lines).rstrip())

    narrative_text = "\n".join(f"[Case]\n{s}" if i == 0 else s for i, s in enumerate(sections)).strip() + "\n"
    return narrative_text, row


def save_narrative(text: str, row: pd.Series, admission_id: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = val_or_na(row.get("newpatient_ID"))
    path = out_dir / f"{pid}_{admission_id}.narrative.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    df_adm, df_bsi = load_tables(data_dir)

    if args.admission_id is not None:
        text, row = build_narrative(args.admission_id, df_adm, df_bsi, args.include_transaminases)
        path = save_narrative(text, row, args.admission_id, out_dir)
        print(f"[INFO] Saved: {path}")
        return

    admissions = sorted(df_adm["admission_ID"].dropna().astype(int).unique())
    if args.max_n:
        admissions = admissions[: args.max_n]
    print(f"[INFO] Generating narratives for {len(admissions)} admissions...")

    n_ok = n_fail = 0
    for i, aid in enumerate(admissions, 1):
        try:
            text, row = build_narrative(int(aid), df_adm, df_bsi, args.include_transaminases)
            save_narrative(text, row, int(aid), out_dir)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[WARN] Failed admission_ID={aid}: {e}")
        if i % 500 == 0:
            print(f"[INFO] Progress: {i}/{len(admissions)} (ok={n_ok}, fail={n_fail})")

    print(f"[INFO] Done. ok={n_ok}, fail={n_fail}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
