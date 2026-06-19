#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
embed_narratives.py

Embeds *.narrative.md files using the OpenAI embeddings API for phenotyping analysis.
Supports sharded saving (one .npy per shard), resumable runs, and end-of-job QC.

Outputs (in --out-dir):
  - embeddings_shards/
      shard_00000.npy
      shard_00001.npy
      ...
  - narrative_embeddings_metadata.csv
      columns: patient_ID, admission_ID, narrative_path, model, shard_id, row_in_shard

Usage:
  export OPENAI_API_KEY="sk-..."
  python embed_narratives.py \\
    --narrative-dir ./narratives \\
    --out-dir ./embeddings_out \\
    --model text-embedding-3-large \\
    --batch-size 64 \\
    --shard-size 2048 \\
    --resume \\
    --qc
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 64
DEFAULT_SHARD_SIZE = 2048

FNAME_RE = re.compile(r"^(?P<patient>.+)_(?P<admission>\d+)\.narrative\.md$")


def parse_args():
    p = argparse.ArgumentParser(description="Embed BSI case narrative files using OpenAI embeddings.")
    p.add_argument("--narrative-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    p.add_argument("--max-narratives", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--base-sleep", type=float, default=1.0)
    p.add_argument("--qc", action="store_true")
    return p.parse_args()


def extract_ids(path: Path):
    m = FNAME_RE.match(path.name)
    return (m.group("patient"), m.group("admission")) if m else ("NA", "NA")


def load_metadata(meta_csv: Path) -> pd.DataFrame:
    if not meta_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(meta_csv)


def infer_next_position(meta_df: pd.DataFrame):
    if meta_df.empty or not {"shard_id", "row_in_shard"}.issubset(meta_df.columns):
        return 0, 0
    last = meta_df.dropna(subset=["shard_id", "row_in_shard"]).sort_values(
        ["shard_id", "row_in_shard"]
    ).iloc[-1]
    return int(last["shard_id"]), int(last["row_in_shard"]) + 1


def embed_with_retry(client, model, texts, max_retries, base_sleep):
    for attempt in range(max_retries + 1):
        try:
            return client.embeddings.create(model=model, input=texts)
        except Exception as e:
            if attempt >= max_retries:
                raise RuntimeError(f"Embedding failed after {max_retries} retries: {e}") from e
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.25))
            print(f"[WARN] Retry {attempt+1}: {e}", file=sys.stderr)


def flush_shard(shard_dir, shard_id, shard_embs, shard_meta, meta_csv):
    path = shard_dir / f"shard_{shard_id:05d}.npy"
    np.save(path, np.stack(shard_embs).astype(np.float32))
    df = pd.DataFrame(shard_meta)
    df.to_csv(meta_csv, mode="a", index=False, header=not meta_csv.exists())
    print(f"[INFO] Saved shard {shard_id:05d}: {path} (n={len(shard_embs)})")


def run_qc(meta_csv, shard_dir, narrative_dir):
    print("[QC] Running checks...")
    meta_df = load_metadata(meta_csv)
    if meta_df.empty:
        print("[QC][WARN] Metadata empty.", file=sys.stderr)
        return
    total_shard_rows = sum(
        int(np.load(sp, mmap_mode="r").shape[0]) for sp in sorted(shard_dir.glob("shard_*.npy"))
    )
    if len(meta_df) != total_shard_rows:
        print(f"[QC][WARN] metadata={len(meta_df)} vs shard_rows={total_shard_rows}", file=sys.stderr)
    else:
        print(f"[QC] Row count OK: {len(meta_df)}")
    n_dup = meta_df["narrative_path"].astype(str).duplicated().sum() if "narrative_path" in meta_df.columns else 0
    if n_dup:
        print(f"[QC][WARN] Duplicate narrative_path entries: {n_dup}", file=sys.stderr)


def main():
    args = parse_args()
    narrative_dir = Path(args.narrative_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "embeddings_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "narrative_embeddings_metadata.csv"

    meta_df = load_metadata(meta_csv) if args.resume else pd.DataFrame()
    done = set(meta_df["narrative_path"].astype(str)) if (args.resume and "narrative_path" in meta_df.columns) else set()

    paths = sorted(narrative_dir.glob("*.narrative.md"))
    if args.max_narratives:
        paths = paths[: args.max_narratives]
    if args.resume:
        paths = [p for p in paths if str(p) not in done]

    if not paths:
        print("[ERROR] No .narrative.md files to embed.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] To embed: {len(paths)}")
    client = OpenAI()
    shard_id, next_row = infer_next_position(meta_df) if args.resume else (0, 0)
    if args.resume and next_row > 0:
        shard_id += 1
        next_row = 0

    shard_embs, shard_meta = [], []
    batch_texts, batch_meta = [], []
    total_ok = 0
    t0 = time.time()

    def process_batch():
        nonlocal shard_id, total_ok
        resp = embed_with_retry(client, args.model, batch_texts, args.max_retries, args.base_sleep)
        for emb_data, meta in zip(resp.data, batch_meta):
            emb = np.array(emb_data.embedding, dtype=np.float32)
            shard_embs.append(emb)
            shard_meta.append({**meta, "shard_id": shard_id, "row_in_shard": len(shard_embs) - 1})
            total_ok += 1
            if len(shard_embs) >= args.shard_size:
                flush_shard(shard_dir, shard_id, shard_embs[:], shard_meta[:], meta_csv)
                shard_id += 1
                shard_embs.clear()
                shard_meta.clear()
        batch_texts.clear()
        batch_meta.clear()

    for i, path in enumerate(paths, 1):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Cannot read {path}: {e}", file=sys.stderr)
            continue
        patient_id, admission_id = extract_ids(path)
        batch_texts.append(text)
        batch_meta.append({"patient_ID": patient_id, "admission_ID": admission_id,
                            "narrative_path": str(path), "model": args.model})
        if len(batch_texts) >= args.batch_size:
            process_batch()
        if i % 500 == 0:
            print(f"[INFO] {i}/{len(paths)} scanned, embedded={total_ok}, "
                  f"elapsed={( time.time()-t0)/60:.1f} min")

    if batch_texts:
        process_batch()
    if shard_embs:
        flush_shard(shard_dir, shard_id, shard_embs, shard_meta, meta_csv)

    print(f"[INFO] Done. embedded={total_ok}, elapsed={(time.time()-t0)/60:.1f} min")
    if args.qc:
        run_qc(meta_csv, shard_dir, narrative_dir)


if __name__ == "__main__":
    main()
