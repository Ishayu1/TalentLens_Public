# TalentLens Pipeline

Data flow across 3 pipeline notebooks → Streamlit app.

```mermaid
flowchart TB
    subgraph INPUTS["Raw Data Sources"]
        direction TB
        A1["DS3 PDFs"]
        A2["Training / Kaggle Resumes"]
        A3["DS3 Metadata"]
    end

    subgraph NB1["01_text_extraction.ipynb"]
        direction TB
        B1["Extract + Clean Resume Text"]
        B2["Filter + Enrich Metadata"]
        B1 --> B2
    end

    A1 --> B1
    A2 --> B1
    A3 --> B2

    subgraph OUT1["Notebook 01 Output"]
        C1["data/processed/<br/>resumes_extracted.json"]
    end

    B2 --> C1

    subgraph NB2["02_embeddings.ipynb"]
        direction TB
        D1["Generate Resume Embeddings"]
    end

    C1 --> D1

    subgraph OUT2["Notebook 02 Outputs"]
        direction LR
        E1["embeddings.npy"]
        E2["resumes_with_embeddings.json"]
    end

    D1 --> E1
    D1 --> E2

    subgraph NB3["03_faiss_indexing.ipynb"]
        direction TB
        F1["Build FAISS Search Artifacts"]
    end

    E1 --> F1
    E2 --> F1
    C1 --> F1

    subgraph OUT3["Notebook 03 Outputs (Project Root)"]
        direction LR
        G1["resume_index.faiss"]
        G2["member_resumes_metadata.json"]
        G3["config.json"]
    end

    F1 --> G1
    F1 --> G2
    F1 --> G3

    subgraph APP["Streamlit App"]
        direction TB
        H1["User Query<br/>(Skills or Job Description)"]
        H2["Semantic Search<br/>(embedding + FAISS + local boosts / filters)"]
        H3["Top Candidate Shortlist"]
        H4["Grok Ranking Layer<br/>(skill extraction for JD,<br/>recruiter assessment,<br/>resume rubric, penalty checks)"]
        H5["Final Ranked Results<br/>+ top-3 explanations"]

        H1 --> H2
        H2 --> H3
        H3 --> H4
        H4 --> H5
    end

    G1 --> H2
    G2 --> H2
    G3 --> H2

    C1 -. fallback resumes .-> H2
    E1 -. fallback embeddings .-> H2
    E2 -. fallback metadata .-> H2

    classDef inputNode fill:#1f3a5f,stroke:#58a6ff,color:#c9d1d9
    classDef processNode fill:#2d1f3a,stroke:#bc8cff,color:#e2d4f0
    classDef fileNode fill:#1a3328,stroke:#3fb950,color:#aff5b4
    classDef appNode fill:#3a2a1a,stroke:#d29922,color:#f0d98c

    class A1,A2,A3 inputNode
    class B1,B2,D1,F1 processNode
    class C1,E1,E2,G1,G2,G3 fileNode
    class H1,H2,H3,H4,H5 appNode
```

## File flow summary

| Stage | Reads From | Writes To |
|-------|------------|-----------|
| **01_text_extraction** | `test/members/`, `test/board/`, `train/`, Kaggle CSV, `members.csv` | `data/processed/resumes_extracted.json` |
| **02_embeddings** | `data/processed/resumes_extracted.json` | `data/processed/embeddings.npy`, `data/processed/resumes_with_embeddings.json` |
| **03_faiss_indexing** | All 3 files in `data/processed/` | `resume_index.faiss`, `member_resumes_metadata.json`, `config.json` (project root) |
| **Streamlit app** | `resume_index.faiss`, `member_resumes_metadata.json`, `config.json` | — |
