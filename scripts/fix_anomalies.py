import os
import shutil
import csv
import re
from pathlib import Path
import fitz

def main():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    test_members = base_dir / "test" / "members"
    report_path = base_dir / "failed_extractions_report.csv"

    def log_failure(name, file_path, reason):
        with open(report_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, str(file_path), reason])

    # 1. Handle member_resume_47.pdf (Canva block)
    f47 = test_members / "member_resume_47.pdf"
    if f47.exists():
        print(f"Removing {f47.name} (Canva blocked)")
        f47.unlink()
        log_failure("Resume 47", "member_resume_47.pdf", "Canva blocked (HTML response)")

    # 2. Handle member_resume_52.pdf (delete page 1)
    f52 = test_members / "member_resume_52.pdf"
    if f52.exists():
        print(f"Fixing {f52.name} (Extracting page 2)")
        try:
            doc = fitz.open(f52)
            if len(doc) > 1:
                doc.select([1])  # Keep only page 2 (0-indexed)
                temp_path = f52.with_suffix('.temp.pdf')
                doc.save(temp_path)
                doc.close()
                shutil.move(temp_path, f52)
                print(f"  Successfully extracted page 2.")
            else:
                print(f"  {f52.name} has only {len(doc)} pages, skipping extraction.")
                doc.close()
        except Exception as e:
            print(f"  Error fixing {f52.name}: {e}")

    # 3. Handle member_resume_anh_long_do
    f_anh = test_members / "member_resume_anh_long_do.pdf"
    if f_anh.exists():
        print(f"Removing blank PDF for Anh Long Do so pdflink placeholder is used.")
        f_anh.unlink()
        log_failure("Anh Long Do", "member_resume_anh_long_do.pdf", "Blank/Image PDF removed")

    # 4. Handle duplicates by NAME
    files = sorted([f for f in test_members.iterdir() if f.is_file()])
    from collections import defaultdict
    name_map = defaultdict(list)
    
    for f in files:
        stem = f.stem
        if stem.startswith('member_resume_'):
            # Strip trailing _1, _2 etc
            core_name = re.sub(r'_[0-9]+$', '', stem[14:])
            # If the format is .txt and there's a .pdf, we should group them differently,
            # but usually it's pdf vs pdf
            name_map[core_name].append(f)

    for core_name, file_list in name_map.items():
        if len(file_list) > 1:
            print(f"Duplicate by name found for: {core_name}")
            # Identify the preferred file:
            # - We prefer .pdf over .txt
            # - We prefer larger file sizes
            
            def get_priority(x):
                # priority: (is_pdf, size)
                return (x.suffix.lower() == '.pdf', x.stat().st_size)
                
            file_list.sort(key=get_priority, reverse=True)
            
            best_file = file_list[0]
            print(f"  Keeping: {best_file.name} ({best_file.stat().st_size} bytes)")
            
            for dup in file_list[1:]:
                print(f"  Deleting duplicate: {dup.name} ({dup.stat().st_size} bytes)")
                dup.unlink()
                log_failure(core_name.replace('_', ' ').title(), dup.name, "Duplicate by name (deleted smaller/older version)")

    print("Finished anomaly and duplicate cleanup.")

if __name__ == "__main__":
    main()
