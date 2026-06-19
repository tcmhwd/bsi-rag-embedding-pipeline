#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
embed_case_reports.py (robust + resumable + sharded saving + QC)

- Reads *.report.md files from --report-dir
- Creates embeddings using OpenAI embedding model
- Saves embeddings into sharded .npy files + appendable metadata CSV

Outputs (in --out-dir):
  - embeddings_shards/
      shard_00000.npy
      shard_00001.npy
      ...
  - case_report_embeddings_metadata.csv
      one row per embedded report, with shard_id + row_in_shard

Resume behavior:
  - --resume: reads metadata CSV and skips already-embedded report_path
  - Also infers "next write position" primarily from metadata (safer than scanning shard files)
  - Optional: --resume-append-last-shard to append to a not-full last shard (default: off)

QC:
  - At end, verifies total rows implied by metadata == total rows across shard npy files
  - Reports duplicates / missing report files (relative to directory scan)

Usage:
  export OPENAI_API_KEY="sk-..."
  python embed_case_reports.py \
    --report-dir ./reports_all \
    --out-dir ./embeddings_out \
    --model text-embedding-3-large \
    --batch-size 64 \
    --shard-size 2048 \
    --resume
"""

import argparse
from pathlib import Path
import sys
import time
import random
import re

import numpy as np
import pandas as pd
from openai import OpenAI


DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 64
DEFAULT_SHARD_SIZE = 2048  # embeddings per shard file


def parse_args():
    p = argparse.ArgumentParser(description="Embed case report .md files using OpenAI embeddings (robust).")
    p.add_argument("--report-dir", type=str, required=True, help="Directory containing *.report.md files")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help=f"Embeddings per shard .npy (default: {DEFAULT_SHARD_SIZE})")
    p.add_argument("--max-reports", type=int, default=None, help="Embed only first N reports (debug).")
    p.add_argument("--resume", action="store_true", help="Resume from existing metadata; skip already-embedded reports.")
    p.add_argument("--resume-append-last-shard", action="store_true",
                   help="If resuming and last shard exists but is not full, append to it (default: off).")
    p.add_argument("--max-retries", type=int, default=8, help="Max retries on API errors (default: 8)")
    p.add_argument("--base-sleep", type=float, default=1.0, help="Base sleep seconds for backoff (default: 1.0)")
    p.add_argument("--qc", action="store_true", help="Run end-of-job QC checks (recommended).")
    return p.parse_args()


FNAME_RE = re.compile(r"^(?P<patient>.+)_(?P<admission>\d+)\.report\.md$")


def extract_ids_from_filename(path: Path):
    """
    Expected: {patient_pseudo_id}_{admission_ID}.report.md
    Example: patient_7_167383899.report.md
    """
    m = FNAME_RE.match(path.name)
    if not m:
        return "NA", "NA"
    return m.group("patient"), m.group("admission")


def load_report_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_metadata(meta_csv: Path) -> pd.DataFrame:
    if not meta_csv.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(meta_csv)
    except Exception:
        # corrupted metadata is dangerous; fail fast
        raise RuntimeError(f"Metadata CSV exists but cannot be read: {meta_csv}")


def load_already_done(meta_df: pd.DataFrame) -> set:
    if meta_df.empty or "report_path" not in meta_df.columns:
        return set()
    return set(meta_df["report_path"].astype(str).tolist())


def infer_next_position(meta_df: pd.DataFrame):
    """
    Infer where to write next based on metadata CSV (preferred).
    Returns (next_shard_id, next_row_in_shard).
    """
    if meta_df.empty:
        return 0, 0

    required = {"shard_id", "row_in_shard"}
    if not required.issubset(set(meta_df.columns)):
        # If old metadata lacks these fields, fall back to start new shards
        return 0, 0

    # last embedded row by (shard_id, row_in_shard)
    meta_df2 = meta_df.dropna(subset=["shard_id", "row_in_shard"]).copy()
    if meta_df2.empty:
        return 0, 0
    meta_df2["shard_id"] = meta_df2["shard_id"].astype(int)
    meta_df2["row_in_shard"] = meta_df2["row_in_shard"].astype(int)

    last = meta_df2.sort_values(["shard_id", "row_in_shard"]).iloc[-1]
    last_shard = int(last["shard_id"])
    last_row = int(last["row_in_shard"])
    # next position is last_row + 1
    return last_shard, last_row + 1


def embed_with_retry(client: OpenAI, model: str, texts: list[str], max_retries: int, base_sleep: float):
    """
    Retry on transient errors (429/5xx/timeouts).
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return resp
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.25)
            print(f"[WARN] Embedding API error (attempt {attempt+1}/{max_retries+1}): {e}", file=sys.stderr)
            if attempt >= max_retries:
                break
            time.sleep(sleep_s)
    raise RuntimeError(f"Embedding failed after retries. Last error: {last_err}")


def save_shard(shard_dir: Path, shard_id: int, embs: np.ndarray) -> Path:
    """
    Save embeddings array (n, dim) float32 to one .npy shard.
    """
    out = shard_dir / f"shard_{shard_id:05d}.npy"
    np.save(out, embs.astype(np.float32))
    return out


def append_metadata(meta_csv: Path, rows: list[dict]):
    df = pd.DataFrame(rows)
    write_header = not meta_csv.exists()
    df.to_csv(meta_csv, mode="a", index=False, header=write_header)


def load_existing_shard(shard_path: Path) -> np.ndarray:
    return np.load(shard_path)


def qc_check(meta_csv: Path, shard_dir: Path, report_dir: Path):
    """
    QC:
      - meta rows == total rows in shards
      - duplicates in metadata (report_path)
      - missing / extra files between report_dir scan and metadata
    """
    print("[QC] Running QC checks...")

    meta_df = load_metadata(meta_csv)
    if meta_df.empty:
        print("[QC][WARN] Metadata is empty. Nothing to QC.", file=sys.stderr)
        return

    # duplicates
    if "report_path" in meta_df.columns:
        dup = meta_df["report_path"].astype(str).duplicated().sum()
        if dup > 0:
            print(f"[QC][WARN] Duplicate report_path rows in metadata: {dup}", file=sys.stderr)

    # shard row counts
    shard_paths = sorted(shard_dir.glob("shard_*.npy"))
    total_rows_shards = 0
    for sp in shard_paths:
        try:
            arr = np.load(sp, mmap_mode="r")
            total_rows_shards += int(arr.shape[0])
        except Exception as e:
            print(f"[QC][ERROR] Failed to load shard {sp}: {e}", file=sys.stderr)

    total_rows_meta = int(len(meta_df))
    if total_rows_meta != total_rows_shards:
        print(f"[QC][WARN] Row count mismatch: metadata={total_rows_meta} vs shards_total={total_rows_shards}", file=sys.stderr)
    else:
        print(f"[QC] Row count OK: {total_rows_meta}")

    # coverage: report_dir vs metadata
    report_paths = set(str(p) for p in report_dir.glob("*.report.md"))
    meta_paths = set(meta_df["report_path"].astype(str).tolist()) if "report_path" in meta_df.columns else set()

    missing_in_meta = report_paths - meta_paths
    extra_in_meta = meta_paths - report_paths

    if missing_in_meta:
        print(f"[QC][WARN] Reports present on disk but missing in metadata: {len(missing_in_meta)}", file=sys.stderr)
    else:
        print("[QC] All report files are represented in metadata (or were intentionally skipped).")

    if extra_in_meta:
        print(f"[QC][WARN] Metadata contains report_path not found on disk: {len(extra_in_meta)}", file=sys.stderr)


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    out_dir = Path(args.out_dir)
    model = args.model
    batch_size = args.batch_size
    shard_size = args.shard_size
    max_reports = args.max_reports
    resume = args.resume
    resume_append_last = args.resume_append_last_shard
    max_retries = args.max_retries
    base_sleep = args.base_sleep
    run_qc = args.qc

    if not report_dir.exists():
        print(f"[ERROR] report-dir not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(out_dir)
    shard_dir = out_dir / "embeddings_shards"
    ensure_dir(shard_dir)

    meta_csv = out_dir / "case_report_embeddings_metadata.csv"

    meta_df = load_metadata(meta_csv) if resume else pd.DataFrame()
    done_paths = load_already_done(meta_df) if resume else set()

    report_paths = sorted(report_dir.glob("*.report.md"))
    if max_reports is not None:
        report_paths = report_paths[: int(max_reports)]

    if not report_paths:
        print(f"[ERROR] No *.report.md files found in: {report_dir}", file=sys.stderr)
        sys.exit(1)

    if resume:
        report_paths = [p for p in report_paths if str(p) not in done_paths]

    print(f"[INFO] report-dir: {report_dir}")
    print(f"[INFO] out-dir: {out_dir}")
    print(f"[INFO] model={model}, batch_size={batch_size}, shard_size={shard_size}, resume={resume}, resume_append_last_shard={resume_append_last}")
    print(f"[INFO] To embed: {len(report_paths)} reports")

    client = OpenAI()
    t0 = time.time()

    # Infer next position from metadata
    shard_id, next_row_in_shard = infer_next_position(meta_df) if resume else (0, 0)

    # If not appending last shard, always start a fresh shard at boundary
    if resume and (not resume_append_last) and next_row_in_shard > 0:
        shard_id += 1
        next_row_in_shard = 0

    # If appending, try load last shard and continue if exists and not full
    shard_embs = []
    shard_meta = []
    if resume and resume_append_last and next_row_in_shard > 0:
        last_shard_path = shard_dir / f"shard_{shard_id:05d}.npy"
        if last_shard_path.exists():
            arr = load_existing_shard(last_shard_path)
            if arr.shape[0] != next_row_in_shard:
                print(
                    f"[WARN] last shard rows ({arr.shape[0]}) != next_row_in_shard from metadata ({next_row_in_shard}). "
                    "Starting a new shard to avoid corruption.",
                    file=sys.stderr,
                )
                shard_id += 1
                next_row_in_shard = 0
            else:
                shard_embs = [arr[i] for i in range(arr.shape[0])]  # will re-stack on flush
        else:
            # metadata says last shard exists, but file missing -> safest start new shard
            print("[WARN] Metadata indicates last shard, but shard file missing. Starting new shard.", file=sys.stderr)
            shard_id += 1
            next_row_in_shard = 0

    total_ok = 0
    batch_texts = []
    batch_meta = []

    for i, path in enumerate(report_paths, start=1):
        try:
            text = load_report_text(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}", file=sys.stderr)
            continue

        patient_id, admission_id = extract_ids_from_filename(path)

        batch_texts.append(text)
        batch_meta.append(
            {
                "patient_ID": patient_id,
                "admission_ID": admission_id,
                "report_path": str(path),
                "model": model,
            }
        )

        if len(batch_texts) >= batch_size:
            resp = embed_with_retry(client, model, batch_texts, max_retries, base_sleep)
            batch_embs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]

            for emb, meta in zip(batch_embs, batch_meta):
                row_in_shard = len(shard_embs)
                shard_embs.append(emb)
                shard_meta.append({**meta, "shard_id": shard_id, "row_in_shard": row_in_shard})
                total_ok += 1

                if len(shard_embs) >= shard_size:
                    shard_path = save_shard(shard_dir, shard_id, np.stack(shard_embs, axis=0))
                    append_metadata(meta_csv, shard_meta)
                    print(f"[INFO] Saved shard {shard_id:05d}: {shard_path} (n={len(shard_embs)})")

                    shard_id += 1
                    shard_embs = []
                    shard_meta = []

            batch_texts = []
            batch_meta = []

        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"[INFO] Progress: {i}/{len(report_paths)} scanned, embedded={total_ok}, elapsed={elapsed/60:.1f} min")

    # final batch
    if batch_texts:
        resp = embed_with_retry(client, model, batch_texts, max_retries, base_sleep)
        batch_embs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        for emb, meta in zip(batch_embs, batch_meta):
            row_in_shard = len(shard_embs)
            shard_embs.append(emb)
            shard_meta.append({**meta, "shard_id": shard_id, "row_in_shard": row_in_shard})
            total_ok += 1

    # final shard flush
    if shard_embs:
        shard_path = save_shard(shard_dir, shard_id, np.stack(shard_embs, axis=0))
        append_metadata(meta_csv, shard_meta)
        print(f"[INFO] Saved final shard {shard_id:05d}: {shard_path} (n={len(shard_embs)})")

    elapsed_total = time.time() - t0
    print(f"[INFO] Done. Embedded={total_ok}. Total elapsed={elapsed_total/60:.1f} min")
    print(f"[INFO] Metadata: {meta_csv}")
    print(f"[INFO] Shards dir: {shard_dir}")

    if run_qc:
        qc_check(meta_csv, shard_dir, report_dir)


if __name__ == "__main__":
    main()

