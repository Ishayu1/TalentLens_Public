import fitz
for fname in ["My PDF.pdf", "RAMBLE.pdf"]:
    doc = fitz.open("manually extracted/" + fname)
    print("-----", fname, "-----")
    print(doc[0].get_text()[:400])
