# 01 — Text Extraction Pipeline Flow

Mermaid diagram for `01_text_extraction.ipynb`: from raw sources to `resumes_extracted.json`.

```mermaid
flowchart TB
    subgraph INPUTS["Inputs"]
        A1["test/members/"]
        A2["test/board/"]
        A3["train/"]
        A4["Resume.csv (Kaggle)"]
        A5["members.csv (DS3)"]
    end

    subgraph PROCESS["process_folder() for each folder"]
        P1["List .pdf, .jpg, .png, .webp, .txt"]
        P2["extract_text() → route by extension"]
        P3["pdfplumber / pytesseract / read .txt"]
        P4["clean_text()"]
        P5["Filter if len < 100 chars"]
        P6["Record: filename, source, text, file_path, word_count"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph KAGGLE["Kaggle CSV path"]
        K1["Read Resume_str column"]
        K2["clean_text() + filter len >= 100"]
        K3["kaggle_records"]
        K1 --> K2 --> K3
    end

    subgraph COMBINE["Combine & Enrich"]
        C1["ds3_records = members + board"]
        C2["Enrich DS3 with members.csv metadata"]
        C3["all_records = ds3 + train + kaggle"]
        C4["Write JSON"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph OUTPUT["Output"]
        O1["data/processed/resumes_extracted.json"]
    end

    A1 --> P_M["process_folder(members)"]
    A2 --> P_B["process_folder(board)"]
    A3 --> P_T["process_folder(train)"]
    A4 --> KAGGLE
    A5 --> C2

    P_M --> R1["ds3_member_records"]
    P_B --> R2["ds3_board_records"]
    P_T --> R3["train_records"]

    R1 --> C1
    R2 --> C1
    R3 --> C3
    K3 --> C3
    C2 --> C3
    C4 --> O1
```

## Summary

The **Process** box above is the pipeline inside each `process_folder()` call (used for members, board, and train).

| Step | What happens |
|------|----------------|
| **Extract** | `extract_text()` routes by extension → PDF (pdfplumber), images (OCR), or plain text. |
| **Clean** | `clean_text()` normalizes line endings, collapses whitespace, limits blank lines, strips non-printable ASCII. |
| **Filter** | Discard documents with cleaned text shorter than `MIN_TEXT_LENGTH` (100 chars). |
| **Folder run** | `process_folder(folder, source)` runs extract → clean → filter and returns list of records. |
| **Sources** | DS3 members + board, train folder, Kaggle CSV (Resume_str) each produce a record list. |
| **Enrich** | DS3 records are matched to `members.csv` and enriched with metadata. |
| **Merge** | All record lists are concatenated and written to `data/processed/resumes_extracted.json`. |
