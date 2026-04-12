import os
import json
import random
from pathlib import Path
import sys
from datetime import datetime

# Force CPU training — avoid MPS OOM on Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
# Override MPS so everything stays on CPU
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_built = lambda: False  # type: ignore

from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample, losses
# from sentence_transformers.evaluation import BinaryClassificationEvaluator

# Force project root
PROJECT_ROOT = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ui"))

from config import SKILL_SUGGESTIONS
from job_description import SKILL_ALIASES

DEVICE = "cpu"
print(f"Training device: {DEVICE}")

print("1. Loading Existing Artifacts")
DATA_DIR = PROJECT_ROOT / "data" / "processed"

with open(DATA_DIR / "resumes_parsed.json", encoding="utf-8") as f:
    resumes_parsed = json.load(f)

with open(DATA_DIR / "resume_chunks.json", encoding="utf-8") as f:
    resume_chunks = json.load(f)

print(f"Loaded {len(resumes_parsed)} parsed resumes and {len(resume_chunks)} chunks.")

print("1.5 Loading training pool texts")
train_texts_path = DATA_DIR / "train_resume_texts.json"
if train_texts_path.exists():
    with open(train_texts_path, "r", encoding="utf-8") as f:
        train_resume_texts = json.load(f)
    print(f"Loaded {len(train_resume_texts)} train texts from cache.")
else:
    train_resume_texts = []

print("2. Building Training Pairs")
chunks_by_candidate = {}
for chunk in resume_chunks:
    cid = chunk.get("candidate_id")
    if not cid:
        continue
    chunks_by_candidate.setdefault(cid, []).append(chunk)

all_candidate_ids = list(chunks_by_candidate.keys())

def extract_matched_skills(text, query_skills):
    haystack = f" {text.lower()} "
    matches = []
    for skill in query_skills:
        aliases = SKILL_ALIASES.get(skill, (skill.lower(),))
        for alias in aliases:
            alias_norm = alias.lower().strip()
            if not alias_norm:
                continue
            if f" {alias_norm} " in haystack or alias_norm in haystack:
                matches.append(skill)
                break
    return matches

pairs = []
random.seed(42)

for resume in resumes_parsed:
    cid = resume.get("candidate_id")
    if not cid or cid not in chunks_by_candidate:
        continue

    skills = resume.get("skills", [])
    if not skills:
        continue

    for _ in range(min(2, len(skills))):
        query_skills = random.sample(skills, min(random.randint(2, 4), len(skills)))
        query = ", ".join(query_skills)

        candidate_chunks = chunks_by_candidate[cid]
        valid_positives = [
            c for c in candidate_chunks
            if c.get("section_type") in {"experience", "projects", "skills"}
            and extract_matched_skills(c.get("text", ""), query_skills)
        ]
        if not valid_positives:
            continue

        pos_chunk = random.choice(valid_positives)
        pos_text = (pos_chunk.get("text", "") or "").strip()[:512]
        if query.strip() and pos_text:
            pairs.append(InputExample(texts=[query, pos_text], label=1.0))

        # 1 hard negative from other DS3 candidates
        for _ in range(20):
            neg_cid = random.choice(all_candidate_ids)
            if neg_cid == cid:
                continue
            neg_chunks = [c for c in chunks_by_candidate[neg_cid] if c.get("section_type") in {"experience", "projects"}]
            if not neg_chunks:
                continue
            neg_chunk = random.choice(neg_chunks)
            neg_text = (neg_chunk.get("text", "") or "").strip()[:512]
            if neg_text and not extract_matched_skills(neg_text, query_skills):
                pairs.append(InputExample(texts=[query, neg_text], label=0.0))
                break

        # 3 easy negatives from train pool
        for _ in range(3):
            if train_resume_texts:
                neg = random.choice(train_resume_texts)
                neg_text = (neg.get("text", "") or "").strip()[:512]
                if neg_text:
                    pairs.append(InputExample(texts=[query, neg_text], label=0.0))

print(f"Generated {len(pairs)} synthetic training pairs.")

print("3. Train/Validation Split")
random.shuffle(pairs)
split_idx = int(len(pairs) * 0.9)
train_samples = pairs[:split_idx]
val_samples = pairs[split_idx:]
print(f"Train size: {len(train_samples)}, Validation size: {len(val_samples)}")

# ── Training with SentenceTransformers ──────────────────────────────────────
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
model = CrossEncoder(MODEL_NAME, num_labels=1, device=DEVICE)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

# No evaluator to avoid compatibility issues in this environment
evaluator = None

output_path = str(PROJECT_ROOT / "models" / "talentlens-cross-encoder-sft-v1")

print("4. Starting Fit …")
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=1,
    evaluation_steps=0,
    warmup_steps=100,
    output_path=output_path,
    optimizer_params={'lr': 2e-5},
    use_amp=False # Disable AMP on CPU to avoid NaNs
)

print(f"Model saved to {output_path}")
