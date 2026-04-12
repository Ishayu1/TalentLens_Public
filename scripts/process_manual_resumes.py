import os
import csv
import string
import shutil
import fitz  # PyMuPDF
import difflib
from pathlib import Path

def clean_filename(name):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned = ''.join(c for c in name if c in valid_chars)
    return cleaned.lower().replace(' ', '_')

def normalize_nospace(text):
    """Lowercases and removes ALL non-alphanumeric chars (including spaces)."""
    return ''.join(c for c in text.lower() if c.isalnum())

def process_manual():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    manual_dir = base_dir / "manually extracted"
    test_members = base_dir / "test" / "members"
    csv_path = base_dir / "Supabase Snippet Talent Pool Resume Links.csv"
    
    if not manual_dir.exists():
        print(f"[ERROR] Directory {manual_dir} does not exist.")
        return

    # Load candidates
    candidates = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('full_name', '').strip()
            if name:
                candidates.append(name)
                
    # Normalize candidates for comparison
    cand_map = {normalize_nospace(c): c for c in candidates}
    
    unmatched = []

    print(f"Scanning {len(list(manual_dir.glob('*.pdf')))} files in {manual_dir.name}...")

    for pdf_path in manual_dir.glob("*.pdf"):
        text = ""
        try:
            doc = fitz.open(pdf_path)
            # Only grab first 2 pages to constrain search space and speed up
            for i in range(min(2, len(doc))):
                text += doc[i].get_text() + " "
            doc.close()
        except:
            pass
            
        norm_text = normalize_nospace(text)
        norm_filename = normalize_nospace(pdf_path.stem)
        
        best_match = None
        best_score = 0
        
        # 1. First, check if exact name is inside the filename
        for norm_c, orig_c in cand_map.items():
            if norm_c and norm_c in norm_filename:
                best_match = orig_c
                best_score = 100
                break
                
        # 2. If not found in filename, check if exact name is in the text
        if not best_match:
            for norm_c, orig_c in cand_map.items():
                if norm_c and len(norm_c) > 4 and norm_c in norm_text:
                    best_match = orig_c
                    best_score = 90
                    break
                    
        # 3. If still not found, do fuzzy matching on filename
        if not best_match:
            for norm_c, orig_c in cand_map.items():
                score = difflib.SequenceMatcher(None, norm_c, norm_filename).ratio()
                if score > 0.8 and score > best_score:
                    best_score = int(score * 100)
                    best_match = orig_c

        if best_match:
            print(f"✅ Match: {pdf_path.name} -> {best_match} (Score: {best_score})")
            slug = clean_filename(best_match)
            target = test_members / f"member_resume_{slug}.pdf"
            
            # Duplicates: Usually manual extractions are meant to fix failures.
            # We keep the manually extracted one, overwriting whatever is there.
            if target.exists():
                print(f"    [!] Overwriting existing {target.name} with manual download.")
                target.unlink()
                
            txt_target = test_members / f"member_resume_{slug}.txt"
            if txt_target.exists():
                txt_target.unlink()
                
            # Shift it
            shutil.move(str(pdf_path), str(target))
        else:
            print(f"❌ NO MATCH: {pdf_path.name}")
            unmatched.append(pdf_path.name)
            
    print(f"\nProcess complete. Unmatched: {len(unmatched)}")
    for u in unmatched:
        print(f"  - {u}")

if __name__ == "__main__":
    process_manual()
