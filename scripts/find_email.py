import fitz
import re

doc = fitz.open("manually extracted/RAMBLE.pdf")
text = ""
for page in doc:
    text += page.get_text()

emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
print("Emails found in RAMBLE.pdf:", emails)
