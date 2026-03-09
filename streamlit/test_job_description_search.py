#!/usr/bin/env python3
"""
Quick test: paste a job description → embed → query FAISS → print top chunk hits.

Run from project root (TalentLens_Public/):
  python streamlit/test_job_description_search.py

Prerequisites:
  - Pipeline 04_2 run: resume_index.faiss and member_chunks_metadata.json exist.
  - sentence-transformers and faiss installed.
"""
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAISS_PATH = PROJECT_ROOT / "resume_index.faiss"
META_PATH = PROJECT_ROOT / "member_chunks_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10


def main():
    if not FAISS_PATH.exists():
        print(f"Missing FAISS index: {FAISS_PATH}")
        print("Run pipeline/04_2_faiss_indexing.ipynb first.")
        sys.exit(1)
    if not META_PATH.exists():
        print(f"Missing metadata: {META_PATH}")
        print("Run pipeline/04_2_faiss_indexing.ipynb first.")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    job_description = input("Paste job description (or press Enter for a sample): ").strip()
    if not job_description:
        job_description = (
            "We are looking for a Software Engineer with 2+ years of experience in Python, "
            "machine learning, and SQL. Experience with PyTorch or TensorFlow is a plus. "
            "BS in Computer Science or related field required."
        )
        print("Using sample job description.\n")

    print("Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if index.ntotal != len(meta):
        print(f"Warning: index has {index.ntotal} vectors but metadata has {len(meta)} rows.")

    print("Embedding job description and searching...")
    q = model.encode([job_description], normalize_embeddings=True).astype("float32")
    faiss.normalize_L2(q)
    scores, indices = index.search(q, TOP_K)

    print(f"\nTop {TOP_K} chunk hits (cosine similarity):\n")
    for i, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        if idx < 0:
            continue
        row = meta[idx]
        text = (row.get("text", "") or "")[:200].replace("\n", " ")
        print(f"  {i}. score={score:.4f}  candidate={row.get('candidate_id', '')}  section={row.get('section_type', '')}")
        print(f"      {text}...")
        print()
    print("Job description embedding + FAISS query works.")


if __name__ == "__main__":
    main()
