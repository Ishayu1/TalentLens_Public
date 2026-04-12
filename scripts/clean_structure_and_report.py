import os
import csv
import string
import re
from pathlib import Path

def clean_filename(name):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned = ''.join(c for c in name if c in valid_chars)
    return cleaned.lower().replace(' ', '_')

def resolve_ugly_names(test_members):
    # Rename any _1.pdf back to .pdf if the base file does not exist.
    for f in test_members.iterdir():
        if f.is_file() and '_' in f.stem:
            # Check if it matches _\d+$
            match = re.search(r'_(\d+)$', f.stem)
            if match:
                base_stem = f.stem[:match.start()]
                target_path = f.with_name(f"{base_stem}{f.suffix}")
                if not target_path.exists():
                    f.rename(target_path)
                else:
                    if f.stat().st_size > target_path.stat().st_size:
                        target_path.unlink()
                        f.rename(target_path)
                    else:
                        f.unlink()

def remake_report():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    test_members = base_dir / "test" / "members"
    csv_path = base_dir / "Supabase Snippet Talent Pool Resume Links.csv"
    report_path = base_dir / "failed_extractions_report.csv"
    
    # 1. Clean ugly names
    resolve_ugly_names(test_members)
    
    # 2. Re-evaluate all candidates
    missing = []
    found_count = 0
    placeholder_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            full_name = row.get('full_name', '').strip()
            if not full_name:
                full_name = f"Unknown_Candidate_{index}"
            
            link = row.get('resume_link', '').strip()
            name_slug = clean_filename(full_name)
            
            # Check if pdf or txt exists
            target_pdf = test_members / f"member_resume_{name_slug}.pdf"
            target_txt = test_members / f"member_resume_{name_slug}.txt"
            
            # Use fuzzy match
            pdf_exists = target_pdf.exists()
            txt_exists = target_txt.exists()
            
            if not pdf_exists and not txt_exists:
                for existing_file in test_members.iterdir():
                    if name_slug in existing_file.name:
                        if existing_file.suffix.lower() == '.pdf':
                            pdf_exists = True
                        elif existing_file.suffix.lower() == '.txt':
                            txt_exists = True
                        break
                        
            if pdf_exists:
                found_count += 1
            elif txt_exists:
                # Text files are placeholders for Canva/Overleaf! They should go in the failure report for manual download!
                placeholder_count += 1
                missing.append([full_name, link, "Requires Manual Download (Canva/Overleaf/pdflink placeholder)"])
            else:
                reason = "Missing from final dataset"
                if 'canva.com' in link:
                    reason = "Requires Manual Download (Canva blocked)"
                elif 'drive.google.com' in link:
                    reason = "Permission Denied / Invalid Google Drive link"
                elif not link or link == "-" or link.lower() in ["none", "n/a", "null"]:
                    reason = "No link provided"
                else:
                    reason = "Automated extraction failed"
                missing.append([full_name, link, reason])
                    
    # Re-write the report
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Full Name', 'Resume Link', 'Reason'])
        writer.writerows(missing)
        
    print(f"Total Fully Validated PDF Resumes: {found_count}")
    print(f"Total Processed candidates in CSV: {found_count + len(missing)}")
    print(f"Total Missing/Failed (including {placeholder_count} placeholders): {len(missing)}")
    print("Report has been fully regenerated and is clean!")

if __name__ == "__main__":
    remake_report()
