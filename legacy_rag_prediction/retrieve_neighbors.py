#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
retrieve_neighbors.py

Purpose
-------
Embed query JSONL files with the same OpenAI embedding model used for the corpus,
search the prebuilt FAISS indexes, and save retrieval results.

Primary design
--------------
- Query text source: retrieval_text (default)
- Corpus index: built previously by build_corpus_embeddings_and_faiss.py
- Similarity: cosine similarity via L2-normalized vectors + FAISS inner product
- Neighborhood risk:
    1) unweighted mean of neighbor y_true
    2) similarity-weighted mean with negative similarities clipped to zero

Inputs
------
- early/ast FAISS bundle directories from build_corpus_embeddings_and_faiss.py
- query JSONL files
- template_eval_labels.csv for query/corpus outcomes

Outputs
-------
output_dir/
  early_internal/
    query_metadata.csv
    query_embeddings.npy
    retrieval_long.csv
    retrieval_summary.csv
    retrieval_manifest.json
  early_external/
    ...
  ast_internal/
    ...
  ast_external/
    ...
  retrieval_build_summary.json

Requirements
------------
pip install openai faiss-cpu numpy pandas tqdm

Environment
-----------
export OPENAI_API_KEY=...

Usage
-----
python retrieve_neighbors.py \
  --early-index-dir ./rag_index_openai/early \
  --ast-index-dir ./rag_index_openai/ast \
  --early-internal-query ./llm_templates_cleaned/rag_queries_early_internal_temporal_test.jsonl \
  --early-external-query ./llm_templates_cleaned/rag_queries_early_external.jsonl \
  --ast-internal-query ./llm_templates_cleaned/rag_queries_ast_internal_temporal_test.jsonl \
  --ast-external-query ./llm_templates_cleaned/rag_queries_ast_external.jsonl \
  --eval-labels-csv ./llm_templates_cleaned/template_eval_labels.csv \
  --output-dir ./rag_retrieval_openai \
  --embedding-model text-embedding-3-large \
  --top-k 5

Notes
-----
- This script does NOT call an LLM for generation. It only performs retrieval.
- Use the same embedding model here as was used to build the corpus FAISS index.
"""

from __future__ import annotations

import argparse
import json
import math
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
    raise ImportError("faiss import failed. Install faiss-cpu or faiss-gpu first.") from e

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("openai import failed. Install with: pip install openai") from e


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--early-index-dir", required=True, type=str)
    p.add_argument("--ast-index-dir", required=True, type=str)

    p.add_argument("--early-internal-query", required=True, type=str)
    p.add_argument("--early-external-query", required=True, type=str)
    p.add_argument("--ast-internal-query", required=True, type=str)
    p.add_argument("--ast-external-query", required=True, type=str)

    p.add_argument("--eval-labels-csv", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)

    p.add_argument(
        "--embedding-model",
        required=True,
        type=str,
        help="Must match the model used for corpus embeddings",
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
    )
    p.add_argument(
        "--top-k",
        default=5,
        type=int,
    )
    p.add_argument(
        "--max-retries",
        default=6,
        type=int,
    )
    p.add_argument(
        "--initial-retry-sleep",
        default=2.0,
        type=float,
    )
    p.add_argument(
        "--truncate-chars",
        default=0,
        type=int,
        help="If >0, truncate query text to this many characters before embedding",
    )
    p.add_argument(
        "--store-neighbor-text",
        action="store_true",
        help="Store truncated neighbor retrieval text in retrieval_long.csv",
    )
    p.add_argument(
        "--max-store-neighbor-text-chars",
        default=400,
        type=int,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


# ============================================================
# Generic helpers
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


def build_query_metadata_df(records: list[dict[str, Any]], text_field: str, truncate_chars: int) -> pd.DataFrame:
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

        for k in ["hospital_code", "newpatient_ID", "admission_ID", "admission_year", "split"]:
            if k in metadata:
                row[k] = metadata[k]

        rows.append(row)

    df = pd.DataFrame(rows)

    required = ["doc_id", "patient_uid", "timepoint", "split", "text_embedded"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required query metadata fields: {missing_required}")

    if df["doc_id"].isna().any():
        raise ValueError("Some query records have missing doc_id")
    if not df["doc_id"].is_unique:
        dup = df.loc[df["doc_id"].duplicated(keep=False), "doc_id"].tolist()[:10]
        raise ValueError(f"Query doc_id is not unique. Examples: {dup}")

    if (df["text_embedded"].astype(str).str.len() == 0).any():
        empty_docs = df.loc[df["text_embedded"].astype(str).str.len() == 0, "doc_id"].tolist()[:10]
        raise ValueError(f"Some query records have empty embedding text. Examples: {empty_docs}")

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
                f"[WARN] Query embedding batch failed on attempt {attempt + 1}/{max_retries}. "
                f"Sleeping {sleep_sec:.1f}s. Error: {e}",
                file=sys.stderr,
            )
            time.sleep(sleep_sec)

    raise RuntimeError(f"Query embedding batch failed after {max_retries} retries") from last_err


def embed_texts_openai(
    texts: list[str],
    model: str,
    batch_size: int,
    max_retries: int,
    initial_retry_sleep: float,
) -> np.ndarray:
    client = get_openai_client()

    all_vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding query batches"):
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
        raise ValueError(f"Embedding row mismatch: expected {len(texts)}, got {out.shape[0]}")
    return out


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def load_corpus_bundle(index_dir: Path, prefix: str):
    meta_path = index_dir / f"{prefix}_corpus_metadata.csv"
    emb_path = index_dir / f"{prefix}_embeddings.npy"
    index_path = index_dir / f"{prefix}_faiss.index"
    manifest_path = index_dir / f"{prefix}_manifest.json"

    for p in [meta_path, emb_path, index_path, manifest_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing corpus bundle file: {p}")

    meta_df = pd.read_csv(meta_path, low_memory=False)
    emb = np.load(emb_path)
    index = faiss.read_index(str(index_path))

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if len(meta_df) != emb.shape[0]:
        raise ValueError(
            f"Corpus metadata row count and embedding row count do not match for {prefix}: "
            f"{len(meta_df)} vs {emb.shape[0]}"
        )
    if index.ntotal != emb.shape[0]:
        raise ValueError(
            f"FAISS ntotal and embedding row count do not match for {prefix}: "
            f"{index.ntotal} vs {emb.shape[0]}"
        )
    if not meta_df["doc_id"].is_unique:
        raise ValueError(f"{prefix} corpus metadata doc_id is not unique")

    return meta_df, emb, index, manifest


def load_eval_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval labels file: {path}")
    df = pd.read_csv(path, low_memory=False)

    required = ["doc_id", "timepoint", "split", "y_true"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in eval labels: {missing}")

    if not df["doc_id"].is_unique:
        dup = df.loc[df["doc_id"].duplicated(keep=False), "doc_id"].tolist()[:10]
        raise ValueError(f"eval_labels doc_id is not unique. Examples: {dup}")

    return df


def similarity_weighted_risk(similarities: list[float], labels: list[float]) -> float | None:
    sims = np.asarray(similarities, dtype=np.float32)
    ys = np.asarray(labels, dtype=np.float32)

    mask = ~np.isnan(ys)
    sims = sims[mask]
    ys = ys[mask]

    if len(ys) == 0:
        return None

    weights = np.clip(sims, 0.0, None)
    if float(weights.sum()) <= 0:
        return float(ys.mean())
    return float((weights * ys).sum() / weights.sum())


def mean_risk(labels: list[float]) -> float | None:
    ys = np.asarray(labels, dtype=np.float32)
    ys = ys[~np.isnan(ys)]
    if len(ys) == 0:
        return None
    return float(ys.mean())


def stringify_list(values: list[Any], sep: str = "|") -> str:
    return sep.join("" if v is None else str(v) for v in values)


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def run_retrieval_for_one_dataset(
    dataset_name: str,
    query_jsonl: Path,
    corpus_meta_df: pd.DataFrame,
    faiss_index,
    eval_labels_df: pd.DataFrame,
    output_dir: Path,
    embedding_model: str,
    text_field: str,
    batch_size: int,
    top_k: int,
    max_retries: int,
    initial_retry_sleep: float,
    truncate_chars: int,
    store_neighbor_text: bool,
    max_store_neighbor_text_chars: int,
    overwrite: bool,
) -> dict[str, Any]:
    ds_out = output_dir / dataset_name
    if ds_out.exists() and any(ds_out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {ds_out}. "
            f"Use --overwrite to proceed."
        )
    ds_out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading query set: {dataset_name}")
    query_records = read_jsonl(query_jsonl)
    query_df = build_query_metadata_df(
        records=query_records,
        text_field=text_field,
        truncate_chars=truncate_chars,
    )

    # join query labels
    query_df = query_df.merge(
        eval_labels_df[["doc_id", "y_true"]],
        on="doc_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"y_true": "query_y_true"})

    # join corpus labels
    corpus_df = corpus_meta_df.merge(
        eval_labels_df[["doc_id", "y_true"]],
        on="doc_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"y_true": "neighbor_y_true"})

    # embed queries
    texts = query_df["text_embedded"].astype(str).tolist()
    print(f"[INFO] Embedding queries for {dataset_name}: n={len(texts)}")
    q_emb = embed_texts_openai(
        texts=texts,
        model=embedding_model,
        batch_size=batch_size,
        max_retries=max_retries,
        initial_retry_sleep=initial_retry_sleep,
    )
    q_emb = l2_normalize_rows(q_emb).astype(np.float32)

    # search
    print(f"[INFO] Searching FAISS for {dataset_name}, top_k={top_k}")
    D, I = faiss_index.search(q_emb, top_k)

    # save query metadata + embeddings
    query_meta_path = ds_out / "query_metadata.csv"
    query_emb_path = ds_out / "query_embeddings.npy"
    query_df.to_csv(query_meta_path, index=False)
    np.save(query_emb_path, q_emb)

    # build long + summary outputs
    long_rows = []
    summary_rows = []

    for q_idx in range(len(query_df)):
        q = query_df.iloc[q_idx]
        sims = D[q_idx].tolist()
        idxs = I[q_idx].tolist()

        neighbor_doc_ids = []
        neighbor_patient_uids = []
        neighbor_splits = []
        neighbor_labels = []
        neighbor_sims = []

        top1_doc_id = None
        top1_sim = None
        top1_y = None

        for rank, (sim, nn_idx) in enumerate(zip(sims, idxs), start=1):
            if nn_idx < 0:
                continue

            n = corpus_df.iloc[nn_idx]

            neighbor_doc_id = n["doc_id"]
            neighbor_patient_uid = n["patient_uid"]
            neighbor_split = n["split"]
            neighbor_y = n["neighbor_y_true"]

            neighbor_doc_ids.append(neighbor_doc_id)
            neighbor_patient_uids.append(neighbor_patient_uid)
            neighbor_splits.append(neighbor_split)
            neighbor_labels.append(np.nan if pd.isna(neighbor_y) else float(neighbor_y))
            neighbor_sims.append(float(sim))

            row = {
                "query_doc_id": q["doc_id"],
                "query_patient_uid": q["patient_uid"],
                "query_timepoint": q["timepoint"],
                "query_split": q["split"],
                "query_y_true": q["query_y_true"],
                "rank": rank,
                "neighbor_doc_id": neighbor_doc_id,
                "neighbor_patient_uid": neighbor_patient_uid,
                "neighbor_split": neighbor_split,
                "neighbor_y_true": neighbor_y,
                "similarity": float(sim),
            }

            if store_neighbor_text and "text_embedded" in corpus_df.columns:
                row["neighbor_text"] = truncate_text(
                    str(n["text_embedded"]),
                    max_store_neighbor_text_chars,
                )

            long_rows.append(row)

            if rank == 1:
                top1_doc_id = neighbor_doc_id
                top1_sim = float(sim)
                top1_y = neighbor_y

        risk_mean = mean_risk(neighbor_labels)
        risk_weighted = similarity_weighted_risk(neighbor_sims, neighbor_labels)
        n_positive = int(np.nansum(np.asarray(neighbor_labels) == 1)) if len(neighbor_labels) > 0 else 0

        summary_rows.append(
            {
                "query_doc_id": q["doc_id"],
                "query_patient_uid": q["patient_uid"],
                "query_timepoint": q["timepoint"],
                "query_split": q["split"],
                "query_y_true": q["query_y_true"],
                "top_k": top_k,
                "n_neighbors_found": len(neighbor_doc_ids),
                "top1_neighbor_doc_id": top1_doc_id,
                "top1_similarity": top1_sim,
                "top1_neighbor_y_true": top1_y,
                "n_positive_in_topk": n_positive,
                "retrieval_neighborhood_risk_mean": risk_mean,
                "retrieval_neighborhood_risk_weighted": risk_weighted,
                "neighbor_doc_ids": stringify_list(neighbor_doc_ids),
                "neighbor_patient_uids": stringify_list(neighbor_patient_uids),
                "neighbor_splits": stringify_list(neighbor_splits),
                "neighbor_y_true_list": stringify_list(neighbor_labels),
                "neighbor_similarity_list": stringify_list([round(x, 6) for x in neighbor_sims]),
            }
        )

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)

    long_path = ds_out / "retrieval_long.csv"
    summary_path = ds_out / "retrieval_summary.csv"
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    manifest = {
        "dataset_name": dataset_name,
        "input_query_jsonl": str(query_jsonl),
        "n_queries": int(len(query_df)),
        "top_k": int(top_k),
        "embedding_model": embedding_model,
        "text_field": text_field,
        "faiss_metric": "inner_product_on_l2_normalized_vectors",
        "query_split_values": sorted(query_df["split"].dropna().astype(str).unique().tolist()),
        "query_timepoint_values": sorted(query_df["timepoint"].dropna().astype(str).unique().tolist()),
        "query_text_char_len_mean": float(query_df["text_embedded"].astype(str).str.len().mean()),
        "query_text_char_len_median": float(query_df["text_embedded"].astype(str).str.len().median()),
        "query_text_char_len_p95": float(query_df["text_embedded"].astype(str).str.len().quantile(0.95)),
        "files": {
            "query_metadata_csv": str(query_meta_path),
            "query_embeddings_npy": str(query_emb_path),
            "retrieval_long_csv": str(long_path),
            "retrieval_summary_csv": str(summary_path),
        },
    }

    manifest_path = ds_out / "retrieval_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(manifest), f, ensure_ascii=False, indent=2)

    return {
        "query_jsonl": str(query_jsonl),
        "n_queries": int(len(query_df)),
        "top_k": int(top_k),
        "outputs": {
            "query_metadata_csv": str(query_meta_path),
            "query_embeddings_npy": str(query_emb_path),
            "retrieval_long_csv": str(long_path),
            "retrieval_summary_csv": str(summary_path),
            "retrieval_manifest_json": str(manifest_path),
        },
    }


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_labels_df = load_eval_labels(Path(args.eval_labels_csv))

    early_corpus_meta, _, early_index, early_manifest = load_corpus_bundle(
        Path(args.early_index_dir), "early"
    )
    ast_corpus_meta, _, ast_index, ast_manifest = load_corpus_bundle(
        Path(args.ast_index_dir), "ast"
    )

    summary = {
        "embedding_model": args.embedding_model,
        "text_field": args.text_field,
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "truncate_chars": int(args.truncate_chars),
        "early_corpus_manifest_embedding_model": early_manifest.get("embedding_model"),
        "ast_corpus_manifest_embedding_model": ast_manifest.get("embedding_model"),
    }

    # soft warning if model mismatch
    if early_manifest.get("embedding_model") != args.embedding_model:
        print(
            f"[WARN] Query embedding model differs from early corpus manifest model: "
            f"{args.embedding_model} vs {early_manifest.get('embedding_model')}",
            file=sys.stderr,
        )
    if ast_manifest.get("embedding_model") != args.embedding_model:
        print(
            f"[WARN] Query embedding model differs from ast corpus manifest model: "
            f"{args.embedding_model} vs {ast_manifest.get('embedding_model')}",
            file=sys.stderr,
        )

    datasets = [
        ("early_internal", Path(args.early_internal_query), early_corpus_meta, early_index),
        ("early_external", Path(args.early_external_query), early_corpus_meta, early_index),
        ("ast_internal", Path(args.ast_internal_query), ast_corpus_meta, ast_index),
        ("ast_external", Path(args.ast_external_query), ast_corpus_meta, ast_index),
    ]

    for ds_name, ds_query_path, ds_corpus_meta, ds_index in datasets:
        info = run_retrieval_for_one_dataset(
            dataset_name=ds_name,
            query_jsonl=ds_query_path,
            corpus_meta_df=ds_corpus_meta,
            faiss_index=ds_index,
            eval_labels_df=eval_labels_df,
            output_dir=output_dir,
            embedding_model=args.embedding_model,
            text_field=args.text_field,
            batch_size=args.batch_size,
            top_k=args.top_k,
            max_retries=args.max_retries,
            initial_retry_sleep=args.initial_retry_sleep,
            truncate_chars=args.truncate_chars,
            store_neighbor_text=args.store_neighbor_text,
            max_store_neighbor_text_chars=args.max_store_neighbor_text_chars,
            overwrite=args.overwrite,
        )
        summary[ds_name] = info

    summary_path = output_dir / "retrieval_build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(summary), f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")
    print(f"[INFO] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
