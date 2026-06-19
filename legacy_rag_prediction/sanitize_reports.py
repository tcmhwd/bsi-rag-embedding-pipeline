#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanitize_reports.py

Removes the [Outcome] section and any outcome-leaking keywords from case
report Markdown files to prevent label leakage before embedding.

Usage:
  python sanitize_reports.py --in_dir ./reports_all --out_dir ./reports_sanitized
"""
import argparse, re
from pathlib import Path

# Remove the entire [Outcome] section block
OUTCOME_BLOCK = re.compile(r'\n?\[Outcome\][\s\S]*?(?=\n\[[A-Za-z].*?\]|$)', re.IGNORECASE)
# Remove any remaining lines containing outcome-leaking keywords
LEAKY_LINES = re.compile(
    r'(in-?hospital death|30[- ]?day mortality|death date|discharge date|expired|mortality)',
    re.IGNORECASE
)

def strip_outcome(txt: str) -> str:
    t = OUTCOME_BLOCK.sub("", txt)
    # Line-level fallback for outcomes scattered outside a formal section block
    lines = []
    for line in t.splitlines():
        if LEAKY_LINES.search(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"

def main():
    ap = argparse.ArgumentParser(description="Strip outcome information from case report .md files.")
    ap.add_argument("--in_dir", required=True, help="Input directory containing *.report.md files")
    ap.add_argument("--out_dir", required=True, help="Output directory for sanitized reports")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n=0
    for p in sorted(in_dir.glob("*.report.md")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        clean = strip_outcome(txt)
        (out_dir/p.name).write_text(clean, encoding="utf-8")
        n += 1
    print(f"[OK] sanitized {n} files -> {out_dir}")

if __name__ == "__main__":
    main()

