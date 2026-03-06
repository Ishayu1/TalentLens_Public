# TalentLens Pipeline

Data flow across 3 pipeline notebooks → Streamlit app.

```mermaid
flowchart TB
    subgraph INPUTS["Raw Data Sources"]
        direction TB
        A1["test/members/<br/>(DS3 Member PDFs)"]
        A2["test/board/<br/>(DS3 Board PDFs)"]
        A3["train/<br/>(Training PDFs)"]
        A4["data/resume-dataset/Resume/Resume.csv<br/>(Kaggle CSV)"]
        A5["data/ds3/member_resumes/members.csv<br/>(DS3 Metadata)"]
    end

    subgraph NB1["01_text_extraction.ipynb"]
        direction TB
        B1["PDF Extraction<br/>(pdfplumber)"]
        B2["Image OCR<br/>(pytesseract)"]
        B3["Text Cleaning<br/>& Normalization"]
        B4["Kaggle Filtering<br/>(ENG / IT / BIZ-DEV)"]
        B5["Fuzzy-Match DS3<br/>Metadata Enrichment"]
        B1 --> B3
        B2 --> B3
        B4 --> B3
        B3 --> B5
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B4
    A5 --> B5

    subgraph OUT1["Notebook 01 Output"]
        C1["data/processed/<br/>resumes_extracted.json"]
    end

    B5 --> C1

    subgraph NB2["02_embeddings.ipynb"]
        direction TB
        D1["Load SentenceTransformer<br/>all-MiniLM-L6-v2 (384-dim)"]
        D2["Encode Resume Text<br/>(batch=32, L2-normalized)"]
        D1 --> D2
    end

    C1 --> D1

    subgraph OUT2["Notebook 02 Outputs"]
        direction LR
        E1["data/processed/<br/>embeddings.npy<br/>(N × 384 float32)"]
        E2["data/processed/<br/>resumes_with_embeddings.json<br/>(metadata + text_preview)"]
    end

    D2 --> E1
    D2 --> E2

    subgraph NB3["03_faiss_indexing.ipynb"]
        direction TB
        F1["Load & Align Data"]
        F2["Filter to DS3 Members"]
        F3["Build FAISS Index<br/>(IndexFlatIP / cosine sim)"]
        F4["Export Artifacts"]
        F1 --> F2
        F2 --> F3
        F3 --> F4
    end

    E1 --> F1
    E2 --> F1
    C1 --> F1

    subgraph OUT3["Notebook 03 Outputs (Project Root)"]
        direction LR
        G1["resume_index.faiss"]
        G2["member_resumes_metadata.json"]
        G3["config.json<br/>(model_name, dim, count)"]
    end

    F4 --> G1
    F4 --> G2
    F4 --> G3

    subgraph APP["Streamlit App"]
        direction LR
        H1["streamlit/search.py"]
        H2["streamlit/config.py"]
    end

    G1 --> H1
    G2 --> H1
    G3 --> H2

    classDef inputNode fill:#1f3a5f,stroke:#58a6ff,color:#c9d1d9
    classDef processNode fill:#2d1f3a,stroke:#bc8cff,color:#e2d4f0
    classDef fileNode fill:#1a3328,stroke:#3fb950,color:#aff5b4
    classDef appNode fill:#3a2a1a,stroke:#d29922,color:#f0d98c

    class A1,A2,A3,A4,A5 inputNode
    class B1,B2,B3,B4,B5,D1,D2,F1,F2,F3,F4 processNode
    class C1,E1,E2,G1,G2,G3 fileNode
    class H1,H2 appNode
```

## File flow summary

| Stage | Reads From | Writes To |
|-------|------------|-----------|
| **01_text_extraction** | `test/members/`, `test/board/`, `train/`, Kaggle CSV, `members.csv` | `data/processed/resumes_extracted.json` |
| **02_embeddings** | `data/processed/resumes_extracted.json` | `data/processed/embeddings.npy`, `data/processed/resumes_with_embeddings.json` |
| **03_faiss_indexing** | All 3 files in `data/processed/` | `resume_index.faiss`, `member_resumes_metadata.json`, `config.json` (project root) |
| **Streamlit app** | `resume_index.faiss`, `member_resumes_metadata.json`, `config.json` | — |
