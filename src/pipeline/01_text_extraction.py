#!/usr/bin/env python
# coding: utf-8

# # 01 — Text Extraction & Cleaning
# 
# Extract text from resume sources and produce a single clean JSON for the rest of the pipeline.
# 
# **Inputs:**
# | Source      | Location         | Format |
# |-------------|------------------|--------|
# | DS3 Members | `test/members/`  | PDF    |
# | DS3 Board   | `test/board/`    | PDF    |
# | Train Set   | `train/`         | PDF    |
# 
# **Output:** `data/processed/resumes_extracted.json`  
# A list of records, each with `{filename, source, text, file_path, word_count, metadata}`

# In[1]:


import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import pdfplumber
from PIL import Image
import pytesseract

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "resumes_extracted.json"

print(f"Project root : {PROJECT_ROOT}")
print(f"Data dir     : {DATA_DIR}")
print(f"Output path  : {OUTPUT_PATH}")


# ## 1 — Extraction helpers

# In[2]:


import fitz

PDF_OCR_FALLBACK_THRESHOLD = 0  # only try OCR when pdfplumber returns no text at all

_tesseract_available = None
_ocr_fallback_used = []  # per process_folder run: count of PDFs that used OCR fallback

def _check_tesseract() -> bool:
    """Check once if Tesseract is installed; warn if not (avoids spamming per-file)."""
    global _tesseract_available
    if _tesseract_available is None:
        try:
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
            print("  [WARN] Tesseract is not installed or not in your PATH. OCR fallback for image PDFs will be skipped. Install it (e.g. brew install tesseract on macOS) to enable.")
    return _tesseract_available


def extract_text_pdf(file_path: Path) -> str | None:
    """Extract text from a PDF using pdfplumber.

    Returns None for obviously broken/non-PDF files so OCR can be skipped.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages).strip()
    except Exception as e:
        error_text = str(e)
        print(f"  [WARN] PDF extraction failed for {file_path.name}: {error_text}")
        if "No /Root object" in error_text or "Cannot open empty file" in error_text:
            return None
        return ""


def extract_text_pdf_ocr(file_path: Path) -> str:
    """Render PDF pages to images and OCR with pytesseract (for image-only/scanned PDFs)."""
    if not _check_tesseract():
        return ""
    try:
        doc = fitz.open(file_path)
        page_texts = []
        mat = fitz.Matrix(2, 2)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_texts.append(pytesseract.image_to_string(img).strip())
        doc.close()
        return "\n".join(page_texts).strip()
    except Exception as e:
        print(f"  [WARN] PDF OCR failed for {file_path.name}: {e}")
        return ""


def extract_text_image(file_path: Path) -> str:
    """OCR an image file using pytesseract."""
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print(f"  [WARN] OCR failed for {file_path.name}: {e}")
        return ""


def extract_text(file_path: Path, allow_pdf_ocr: bool = False) -> str:
    """Route to the correct extractor based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        text = extract_text_pdf(file_path)
        if text is None:
            return ""
        if allow_pdf_ocr and len(text) <= PDF_OCR_FALLBACK_THRESHOLD:
            ocr_text = extract_text_pdf_ocr(file_path)
            if len(ocr_text) > len(text):
                text = ocr_text
                _ocr_fallback_used.append(1)
        return text
    elif ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return extract_text_image(file_path)
    elif ext == ".txt":
        return file_path.read_text(encoding="utf-8", errors="replace").strip()
    else:
        print(f"  [SKIP] Unsupported extension: {ext} ({file_path.name})")
        return ""


print("Extraction helpers loaded.")


# ## 2 — Text cleaning

# In[3]:


MIN_TEXT_LENGTH = 100  # chars — anything shorter is likely a failed extraction


def clean_text(raw: str) -> str:
    """Normalize whitespace, strip non-printable chars, collapse blank lines."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)          # max 2 consecutive newlines
    text = re.sub(r"[^\x20-\x7E\n]", "", text)      # drop non-printable ASCII
    return text.strip()


print("Cleaning function loaded.")


# ## 3 — Process each data source
# 
# Each helper scans a directory, extracts + cleans text, and returns a list of resume records.

# In[4]:


def _process_one_file(fp: Path, source: str, allow_pdf_ocr: bool = False) -> Dict | None:
    """Extract and clean one file; return record or None if too short."""
    raw = extract_text(fp, allow_pdf_ocr=allow_pdf_ocr)
    text = clean_text(raw)
    if len(text) < MIN_TEXT_LENGTH:
        return None
    return {
        "filename": fp.name,
        "source": source,
        "text": text,
        "file_path": str(fp),
        "word_count": len(text.split()),
    }


def process_folder(folder: Path, source: str, limit: int | None = None, max_workers: int = 1, allow_pdf_ocr: bool = False) -> List[Dict]:
    """Extract text from every supported file in *folder*. Use max_workers > 1 for parallel speedup."""
    if not folder.exists():
        print(f"[SKIP] Folder not found: {folder}")
        return []

    files = sorted(f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".txt"})
    if limit:
        files = files[:limit]

    try:
        _ocr_fallback_used.clear()
    except NameError:
        pass

    if max_workers is not None and max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one_file, fp, source, allow_pdf_ocr): fp for fp in files}
            for future in tqdm(as_completed(futures), total=len(futures), desc=source):
                rec = future.result()
                if rec is not None:
                    records.append(rec)
    else:
        records = []
        for fp in tqdm(files, desc=source):
            rec = _process_one_file(fp, source, allow_pdf_ocr)
            if rec is not None:
                records.append(rec)

    print(f"  -> {len(records)} / {len(files)} files yielded usable text")
    try:
        n = len(_ocr_fallback_used)
        if n:
            print(f"  -> OCR fallback used for {n} PDFs")
        _ocr_fallback_used.clear()
    except NameError:
        pass
    return records


print("Folder processor loaded.")


# ### 3a — Test Set (DS3 Members & Board)
# 
# We process the organized `test/` folder.

# In[5]:


TEST_MEMBERS_DIR = PROJECT_ROOT / "test" / "members"

ds3_member_records = process_folder(TEST_MEMBERS_DIR, source="ds3_members")

ds3_records = ds3_member_records
print(f"Test total: {len(ds3_records)} resumes")


# ### 3b — Train Set
# 
# We process the organized `train/` folder.

# In[6]:


TRAIN_DIR = PROJECT_ROOT / "train"
train_records = process_folder(TRAIN_DIR, source="train", max_workers=8, allow_pdf_ocr=True)


# ### 3c — Enrich DS3 resumes with member metadata
# 
# Join info from `members.csv` (name, major, graduation year, links) onto the DS3 records so it travels with the resume through the rest of the pipeline.

# In[7]:


MEMBERS_CSV = DATA_DIR / "ds3" / "member_resumes" / "members.csv"

members_df = None
if MEMBERS_CSV.exists():
    members_df = pd.read_csv(MEMBERS_CSV)
    print(f"Loaded members.csv: {len(members_df)} rows")
    print(f"Columns: {members_df.columns.tolist()}")
else:
    print(f"[SKIP] members.csv not found at {MEMBERS_CSV}")


# 

# In[8]:


def enrich_ds3_records(records: List[Dict], members_df: pd.DataFrame) -> List[Dict]:
    """Fuzzy-match DS3 resume filenames to rows in members.csv and attach metadata."""
    if members_df is None or members_df.empty:
        return records

    for rec in records:
        stem = Path(rec["filename"]).stem.replace("_", " ").replace("-", " ").lower()
        for _, row in members_df.iterrows():
            name = str(row.get("Full Name", "")).lower()
            if name and name in stem:
                rec["metadata"] = {
                    "full_name": row.get("Full Name", ""),
                    "major": row.get("Major", ""),
                    "graduation_year": str(row.get("Graduation Year", "")),
                    "resume_link": row.get("Resume Link", ""),
                    "linkedin": row.get("Linkedin Link", ""),
                    "github": row.get("Github Link", ""),
                }
                break
        else:
            rec.setdefault("metadata", {})

    matched = sum(1 for r in records if r.get("metadata"))
    print(f"Enriched {matched} / {len(records)} DS3 records with members.csv metadata")
    return records


if members_df is not None:
    ds3_records = enrich_ds3_records(ds3_records, members_df)


# ## 5 — Combine all sources & save

# In[9]:


all_records = ds3_records + train_records


# In[10]:





# 

# In[10]:


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_records, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_records)} records to {OUTPUT_PATH}")
print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


# ## 6 — Quick sanity check
# 
# Preview a few records to make sure extraction looks reasonable.

# In[11]:


for i, rec in enumerate(all_records[:3]):
    print(f"\n{'=' * 60}")
    print(f"[{i}] {rec['filename']}  (source={rec['source']}, words={rec['word_count']})")
    print("-" * 60)
    print(rec["text"][:500])
    print("...")


# In[ ]:





# In[ ]:




