#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
check_repository_consistency.py

Lightweight heuristic consistency checker for the repository.
Checks that the repository aligns with the phenotyping manuscript.

Checks performed:
  1. Required scripts exist.
  2. make_case_narratives.py does not include pseudo-ID in narrative text.
  3. make_case_narratives.py does not include outcome fields in narrative text.
  4. cluster_embeddings_umap_hdbscan.py uses --umap-n-components-cluster 3 as default.
  5. cluster_embeddings_umap_hdbscan.py uses --umap-n-components-viz 2 as default.
  6. README.md does not contain old unqualified RAG wording outside a clearly labeled
     legacy section.
  7. requirements.txt has one package per line (no multi-package lines).
  8. Legacy scripts are in legacy_rag_prediction/ subdirectory.

Run:
  python check_repository_consistency.py --repo-dir .
"""

import argparse
import re
import sys
from pathlib import Path


REQUIRED_SCRIPTS = [
    "make_case_narratives.py",
    "embed_narratives.py",
    "cluster_embeddings_umap_hdbscan.py",
    "structured_variable_clustering_comparators.py",
    "supervised_backmapping.py",
    "phenotype_interpretability.py",
    "mortality_trajectory_analysis.py",
    "ecoli_within_pathogen_analysis.py",
    "make_figures_tables.py",
    "check_repository_consistency.py",
]

LEGACY_SCRIPTS = [
    "make_case_reports.py",
    "embed_case_reports.py",
    "build_corpus_embeddings_and_faiss.py",
    "retrieve_neighbors.py",
    "sanitize_reports.py",
]

# Outcome-related field names that must not appear in narrative body
OUTCOME_FIELDS = [
    "mortality",
    "death_date",
    "discharge_status",
    "length_of_stay",
    "follow_up",
    "time_to_event",
    "cluster_label",
    "inhospital_mortality",
    "mortality_30d",
    "days_to_death",
]

# Phrases that should not appear in README outside a legacy section
PROHIBITED_README_PHRASES = [
    "BSI RAG Embedding Pipeline",
    "Dual-Timepoint Machine Learning",
    "30-Day Mortality Prediction",
    "FAISS index construction",
    "nearest-neighbor retrieval",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Heuristic consistency checker for the BSI phenotyping repository."
    )
    p.add_argument("--repo-dir", type=str, default=".",
                   help="Path to the repository root (default: current directory)")
    return p.parse_args()


def check_required_scripts_exist(repo_dir: Path):
    """Check 1: All required scripts exist in repo_dir."""
    missing = [s for s in REQUIRED_SCRIPTS if not (repo_dir / s).exists()]
    if missing:
        return False, f"Missing required scripts: {missing}"
    return True, f"All {len(REQUIRED_SCRIPTS)} required scripts present."


def check_no_pseudo_id_in_narrative(repo_dir: Path):
    """Check 2: make_case_narratives.py does not include pseudo-ID in narrative text body.

    Checks that no line writing pseudo-ID into the narrative sections still exists.
    """
    path = repo_dir / "make_case_narratives.py"
    if not path.exists():
        return False, "make_case_narratives.py not found."

    text = path.read_text(encoding="utf-8")

    # Look for pattern that puts pseudo-ID into narrative sections (f-string in sections list)
    # The save_narrative function may still reference it for file naming — that's OK.
    # We look for pseudo-ID being written into the sections[] list.
    suspicious_patterns = [
        r"Patient pseudo-ID.*newpatient_ID",
        r"newpatient_ID.*sections\.append",
    ]
    for pat in suspicious_patterns:
        if re.search(pat, text):
            return False, (
                f"Pattern '{pat}' found in make_case_narratives.py. "
                "Pseudo-ID may still be included in narrative text body."
            )

    # Also check that the admin section no longer starts with a pseudo-ID line
    # by looking for the specific f-string pattern in the build_narrative admin block
    if 'f"Patient pseudo-ID: {val_or_na(row.get(\'newpatient_ID\'))}"' in text:
        return False, (
            "Pseudo-ID f-string found in narrative text sections of make_case_narratives.py."
        )

    return True, "No pseudo-ID found in narrative text sections."


def check_no_outcome_fields_in_narrative(repo_dir: Path):
    """Check 3: make_case_narratives.py does not include outcome fields in narrative text.

    Checks that known outcome column names are not added to the sections[] list
    in build_narrative(). Note: they may appear in comments or exclusion lists.
    """
    path = repo_dir / "make_case_narratives.py"
    if not path.exists():
        return False, "make_case_narratives.py not found."

    text = path.read_text(encoding="utf-8")

    # Extract the build_narrative function body
    fn_match = re.search(r"def build_narrative\(.*?\ndef \w", text, re.DOTALL)
    if not fn_match:
        fn_body = text  # fallback: check whole file
    else:
        fn_body = fn_match.group(0)

    issues = []
    for field in OUTCOME_FIELDS:
        # Check if outcome field appears in an f-string within sections.append() context
        # Pattern: field name in a format string that adds to narrative sections
        pattern = rf'f"[^"]*{field}[^"]*"'
        matches = re.findall(pattern, fn_body)
        # Filter out lines that are clearly in comments
        non_comment_matches = [m for m in matches if not m.startswith("#")]
        if non_comment_matches:
            issues.append(f"'{field}' in f-string: {non_comment_matches[:1]}")

    if issues:
        return False, f"Possible outcome fields in narrative text: {issues}"
    return True, "No outcome fields detected in narrative text sections."


def check_umap_cluster_default_3(repo_dir: Path):
    """Check 4: cluster_embeddings_umap_hdbscan.py uses --umap-n-components-cluster 3 as default."""
    path = repo_dir / "cluster_embeddings_umap_hdbscan.py"
    if not path.exists():
        return False, "cluster_embeddings_umap_hdbscan.py not found."

    text = path.read_text(encoding="utf-8")
    # Look for add_argument with umap-n-components-cluster and default=3
    pattern = r"umap-n-components-cluster.*?default=3"
    if re.search(pattern, text, re.DOTALL):
        return True, "--umap-n-components-cluster default=3 found."
    return False, "--umap-n-components-cluster default=3 NOT found in cluster_embeddings_umap_hdbscan.py."


def check_umap_viz_default_2(repo_dir: Path):
    """Check 5: cluster_embeddings_umap_hdbscan.py uses --umap-n-components-viz 2 as default."""
    path = repo_dir / "cluster_embeddings_umap_hdbscan.py"
    if not path.exists():
        return False, "cluster_embeddings_umap_hdbscan.py not found."

    text = path.read_text(encoding="utf-8")
    pattern = r"umap-n-components-viz.*?default=2"
    if re.search(pattern, text, re.DOTALL):
        return True, "--umap-n-components-viz default=2 found."
    return False, "--umap-n-components-viz default=2 NOT found in cluster_embeddings_umap_hdbscan.py."


def _strip_legacy_section(text: str) -> str:
    """Remove the legacy section from README text for phrase checks.

    Heuristic: removes everything after a line containing 'legacy' (case-insensitive)
    or 'Legacy RAG' section header.
    """
    lines = text.split("\n")
    out_lines = []
    in_legacy = False
    for line in lines:
        if re.search(r"(?i)(legacy|legacy_rag_prediction)", line) and line.startswith("#"):
            in_legacy = True
        if not in_legacy:
            out_lines.append(line)
    return "\n".join(out_lines)


def check_readme_no_old_rag_wording(repo_dir: Path):
    """Check 6: README.md does not contain old unqualified RAG wording outside legacy section."""
    path = repo_dir / "README.md"
    if not path.exists():
        return False, "README.md not found."

    text = path.read_text(encoding="utf-8")
    # Strip legacy section before checking
    text_no_legacy = _strip_legacy_section(text)

    found = []
    for phrase in PROHIBITED_README_PHRASES:
        if phrase.lower() in text_no_legacy.lower():
            found.append(phrase)

    if found:
        return False, f"Prohibited phrase(s) found in README.md outside legacy section: {found}"
    return True, "No prohibited RAG wording found in README.md outside legacy section."


def check_requirements_one_per_line(repo_dir: Path):
    """Check 7: requirements.txt has one package per line (no multi-package lines)."""
    path = repo_dir / "requirements.txt"
    if not path.exists():
        return False, "requirements.txt not found."

    text = path.read_text(encoding="utf-8")
    issues = []
    for i, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue
        # Check for multiple packages on one line (e.g., "numpy pandas")
        # Allow version specifiers: >=, <=, ==, ~=, !=
        # Split by whitespace and count package-like tokens
        tokens = [t for t in stripped.split() if not t.startswith("#")]
        package_tokens = [t for t in tokens if re.match(r"^[a-zA-Z]", t)]
        if len(package_tokens) > 1:
            issues.append(f"Line {i}: '{stripped}' (multiple packages?)")

    if issues:
        return False, f"Possible multi-package lines in requirements.txt: {issues}"
    return True, "requirements.txt: one package per line (OK)."


def check_legacy_scripts_in_subdir(repo_dir: Path):
    """Check 8: Legacy scripts are in legacy_rag_prediction/ subdirectory, not repo root."""
    legacy_dir = repo_dir / "legacy_rag_prediction"
    issues = []

    # Check that legacy scripts are NOT in root
    root_found = [s for s in LEGACY_SCRIPTS if (repo_dir / s).exists()]
    if root_found:
        issues.append(f"Legacy scripts found in repo root (should be in legacy_rag_prediction/): {root_found}")

    # Check that legacy scripts ARE in legacy_rag_prediction/
    if not legacy_dir.exists():
        issues.append("legacy_rag_prediction/ directory does not exist.")
    else:
        not_in_subdir = [s for s in LEGACY_SCRIPTS if not (legacy_dir / s).exists()]
        if not_in_subdir:
            issues.append(f"Legacy scripts not found in legacy_rag_prediction/: {not_in_subdir}")

    if issues:
        return False, "; ".join(issues)
    return True, f"All {len(LEGACY_SCRIPTS)} legacy scripts found in legacy_rag_prediction/ (not in root)."


def main():
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    print(f"[INFO] Checking repository: {repo_dir}\n")

    checks = [
        ("1. Required scripts exist", check_required_scripts_exist),
        ("2. No pseudo-ID in narrative text", check_no_pseudo_id_in_narrative),
        ("3. No outcome fields in narrative text", check_no_outcome_fields_in_narrative),
        ("4. UMAP cluster default=3", check_umap_cluster_default_3),
        ("5. UMAP viz default=2", check_umap_viz_default_2),
        ("6. README no old RAG wording", check_readme_no_old_rag_wording),
        ("7. requirements.txt one-per-line", check_requirements_one_per_line),
        ("8. Legacy scripts in subdirectory", check_legacy_scripts_in_subdir),
    ]

    any_failed = False
    results = []
    for name, fn in checks:
        try:
            passed, message = fn(repo_dir)
        except Exception as e:
            passed = False
            message = f"Check raised exception: {e}"
        status = "PASS" if passed else "FAIL"
        if not passed:
            any_failed = True
        results.append((status, name, message))
        print(f"[{status}] {name}")
        print(f"       {message}\n")

    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = sum(1 for r in results if r[0] == "FAIL")
    print(f"--- Summary: {n_pass}/{len(results)} checks passed, {n_fail} failed ---")

    if any_failed:
        print("\n[ERROR] Some checks failed. Resolve failures before creating release.")
        sys.exit(1)
    else:
        print("\n[OK] All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
