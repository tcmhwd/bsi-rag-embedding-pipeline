#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_corpus_embeddings_and_faiss.py

Purpose
-------
Build OpenAI-embedding-based corpus vectors and FAISS indexes for:
- rag_corpus_early_development.jsonl
- rag_corpus_ast_development.jsonl

Primary design
--------------
- Uses retrieval_text (default) as the embedding input
- Builds one FAISS index for early and one for ast
- Uses cosine similarity via:
    1) float32 embedding vectors
    2) L2 normalization
    3) FAISS IndexFlatIP

Outputs
-------
output_dir/
  early/
    early_corpus_metadata.csv
    early_embeddings.npy
    early_faiss.index
    early_manifest.json
  ast/
    ast_corpus_metadata.csv
    ast_embeddings.npy
    ast_faiss.index
    ast_manifest.json
  build_summary.json

Requirements
------------
pip install openai faiss-cpu numpy pandas tqdm

Environment
-----------
export OPENAI_API_KEY=...

Usage
-----
python build_corpus_embeddings_and_faiss.py \
  --early-corpus ./llm_templates_cleaned/rag_corpus_early_development.jsonl \
  --ast-corpus ./llm_templates_cleaned/rag_corpus_ast_development.jsonl \
  --output-dir ./rag_index_openai \
  --embedding-model text-embedding-3-large

Notes
-----
- This script only builds corpus embeddings and FAISS indexes.
- Query embedding / retrieval should be handled in a separate script.
- It is safer to keep corpus building and query retrieval separate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss import failed. Install faiss-cpu or faiss-gpu first."
    ) from e

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "openai import failed. Install with: pip install openai"
    ) from e


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--early-corpus", required=True, type=str)
    p.add_argument("--ast-corpus", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)

    p.add_argument(
        "--embedding-model",
        required=True,
        type=str,
        help="OpenAI embedding model name to use",
    )
    p.add_argument(
        "--text-field",
        default="retrieval_text",
        type=str,
        help="JSONL field to embed. Default: retrieval_text",
    )
    p.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="Number of texts per embedding API call",
    )
    p.add_argument(
        "--max-retries",
        default=6,
        type=int,
        help="Max retries per embedding batch",
    )
    p.add_argument(
        "--initial-retry-sleep",
        default=2.0,
        type=float,
        help="Initial sleep seconds before retry; exponential backoff",
    )
    p.add_argument(
        "--truncate-chars",
        default=0,
        type=int,
        help="If >0, truncate text to this many characters before embedding",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    return p.parse_args()


# ============================================================
# Helpers
# ============================================================
def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if pd.isna(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}") from e

    if len(records) == 0:
        raise ValueError(f"No records found in {path}")

    return records


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return " ".join(s.split())


def maybe_truncate_text(text: str, truncate_chars: int) -> str:
    if truncate_chars and truncate_chars > 0:
        return text[:truncate_chars]
    return text


def build_metadata_df(records: list[dict[str, Any]], text_field: str, truncate_chars: int) -> pd.DataFrame:
    rows = []

    for rec in records:
        metadata = rec.get("metadata", {}) or {}
        text = clean_text(rec.get(text_field, ""))

        row = {
            "doc_id": rec.get("doc_id"),
            "patient_uid": rec.get("patient_uid"),
            "timepoint": rec.get("timepoint"),
            "split": rec.get("split"),
            "text_field": text_field,
            "text_raw_n_chars": len(text),
            "text_embedded": maybe_truncate_text(text, truncate_chars),
        }

        # keep a few metadata columns if available
        for k in ["hospital_code", "newpatient_ID", "admission_ID", "admission_year", "split"]:
            if k in metadata:
                row[k] = metadata[k]

        rows.append(row)

    df = pd.DataFrame(rows)

    required = ["doc_id", "patient_uid", "timepoint", "split", "text_embedded"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required metadata fields: {missing_required}")

    if df["doc_id"].isna().any():
        raise ValueError("Some records have missing doc_id")
    if not df["doc_id"].is_unique:
        dup = df.loc[df["doc_id"].duplicated(keep=False), "doc_id"].tolist()[:10]
        raise ValueError(f"doc_id is not unique. Examples: {dup}")

    if df["text_embedded"].isna().any():
        raise ValueError("Some records have missing embedding text")
    if (df["text_embedded"].astype(str).str.len() == 0).any():
        empty_docs = df.loc[df["text_embedded"].astype(str).str.len() == 0, "doc_id"].tolist()[:10]
        raise ValueError(f"Some records have empty embedding text. Examples: {empty_docs}")

    return df


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def embed_batch_with_retry(
    client: OpenAI,
    texts: list[str],
    model: str,
    max_retries: int,
    initial_retry_sleep: float,
) -> np.ndarray:
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=model,
                input=texts,
            )
            vectors = [item.embedding for item in resp.data]
            arr = np.asarray(vectors, dtype=np.float32)
            return arr

        except Exception as e:
            last_err = e
            sleep_sec = initial_retry_sleep * (2 ** attempt)
            print(
                f"[WARN] Embedding batch failed on attempt {attempt + 1}/{max_retries}. "
                f"Sleeping {sleep_sec:.1f}s. Error: {e}",
                file=sys.stderr,
            )
            time.sleep(sleep_sec)

    raise RuntimeError(f"Embedding batch failed after {max_retries} retries") from last_err


def embed_texts_openai(
    texts: list[str],
    model: str,
    batch_size: int,
    max_retries: int,
    initial_retry_sleep: float,
) -> np.ndarray:
    client = get_openai_client()

    all_vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[start : start + batch_size]
        arr = embed_batch_with_retry(
            client=client,
            texts=batch,
            model=model,
            max_retries=max_retries,
            initial_retry_sleep=initial_retry_sleep,
        )
        all_vecs.append(arr)

    out = np.vstack(all_vecs).astype(np.float32)

    if out.shape[0] != len(texts):
        raise ValueError(
            f"Embedding row mismatch: expected {len(texts)}, got {out.shape[0]}"
        )
    return out


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def build_faiss_ip_index(embeddings: np.ndarray):
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index_bundle(
    name: str,
    meta_df: pd.DataFrame,
    embeddings: np.ndarray,
    index,
    out_dir: Path,
    embedding_model: str,
    text_field: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / f"{name}_corpus_metadata.csv"
    emb_path = out_dir / f"{name}_embeddings.npy"
    index_path = out_dir / f"{name}_faiss.index"
    manifest_path = out_dir / f"{name}_manifest.json"

    meta_df.to_csv(meta_path, index=False)
    np.save(emb_path, embeddings)
    faiss.write_index(index, str(index_path))

    manifest = {
        "name": name,
        "n_docs": int(len(meta_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_model": embedding_model,
        "text_field": text_field,
        "faiss_metric": "inner_product_on_l2_normalized_vectors",
        "doc_id_unique": bool(meta_df["doc_id"].is_unique),
        "timepoint_values": sorted(meta_df["timepoint"].dropna().astype(str).unique().tolist()),
        "split_values": sorted(meta_df["split"].dropna().astype(str).unique().tolist()),
        "text_char_len_mean": float(meta_df["text_embedded"].astype(str).str.len().mean()),
        "text_char_len_median": float(meta_df["text_embedded"].astype(str).str.len().median()),
        "text_char_len_p95": float(meta_df["text_embedded"].astype(str).str.len().quantile(0.95)),
        "files": {
            "metadata_csv": str(meta_path),
            "embeddings_npy": str(emb_path),
            "faiss_index": str(index_path),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(manifest), f, ensure_ascii=False, indent=2)

    return {
        "metadata_csv": str(meta_path),
        "embeddings_npy": str(emb_path),
        "faiss_index": str(index_path),
        "manifest_json": str(manifest_path),
        "n_docs": int(len(meta_df)),
        "embedding_dim": int(embeddings.shape[1]),
    }


def process_one_corpus(
    name: str,
    corpus_path: Path,
    output_dir: Path,
    embedding_model: str,
    text_field: str,
    batch_size: int,
    max_retries: int,
    initial_retry_sleep: float,
    truncate_chars: int,
    overwrite: bool,
) -> dict[str, Any]:
    subdir = output_dir / name
    if subdir.exists() and any(subdir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {subdir}. "
            f"Use --overwrite to proceed."
        )

    print(f"[INFO] Reading {name} corpus: {corpus_path}")
    records = read_jsonl(corpus_path)
    meta_df = build_metadata_df(records, text_field=text_field, truncate_chars=truncate_chars)

    # sanity: corpus is supposed to be development only
    split_values = sorted(meta_df["split"].dropna().astype(str).unique().tolist())
    if split_values != ["development"]:
        print(
            f"[WARN] {name} corpus split values are not exactly ['development']: {split_values}",
            file=sys.stderr,
        )

    texts = meta_df["text_embedded"].astype(str).tolist()

    print(f"[INFO] Embedding {name} corpus: n={len(texts)}")
    embeddings = embed_texts_openai(
        texts=texts,
        model=embedding_model,
        batch_size=batch_size,
        max_retries=max_retries,
        initial_retry_sleep=initial_retry_sleep,
    )

    print(f"[INFO] Normalizing {name} embeddings")
    embeddings = l2_normalize_rows(embeddings).astype(np.float32)

    print(f"[INFO] Building FAISS index for {name}")
    index = build_faiss_ip_index(embeddings)

    print(f"[INFO] Saving {name} bundle")
    saved = save_index_bundle(
        name=name,
        meta_df=meta_df,
        embeddings=embeddings,
        index=index,
        out_dir=subdir,
        embedding_model=embedding_model,
        text_field=text_field,
    )

    return {
        "input_jsonl": str(corpus_path),
        "split_values": split_values,
        "text_field": text_field,
        "saved": saved,
    }


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    early_path = Path(args.early_corpus)
    ast_path = Path(args.ast_corpus)

    summary = {
        "embedding_model": args.embedding_model,
        "text_field": args.text_field,
        "batch_size": int(args.batch_size),
        "max_retries": int(args.max_retries),
        "initial_retry_sleep": float(args.initial_retry_sleep),
        "truncate_chars": int(args.truncate_chars),
        "output_dir": str(output_dir),
    }

    early_info = process_one_corpus(
        name="early",
        corpus_path=early_path,
        output_dir=output_dir,
        embedding_model=args.embedding_model,
        text_field=args.text_field,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        initial_retry_sleep=args.initial_retry_sleep,
        truncate_chars=args.truncate_chars,
        overwrite=args.overwrite,
    )
    ast_info = process_one_corpus(
        name="ast",
        corpus_path=ast_path,
        output_dir=output_dir,
        embedding_model=args.embedding_model,
        text_field=args.text_field,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        initial_retry_sleep=args.initial_retry_sleep,
        truncate_chars=args.truncate_chars,
        overwrite=args.overwrite,
    )

    summary["early"] = early_info
    summary["ast"] = ast_info

    summary_path = output_dir / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(summary), f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")
    print(f"[INFO] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
