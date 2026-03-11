import os
import img2pdf
from pathlib import Path

files_to_fix = [
    "data/ds3/board_resumes/board_resume_50.pdf",
    "data/ds3/member_resumes/member_resume_daniel_vinh_vuong.pdf"
]

project_root = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")

for rel_path in files_to_fix:
    abs_path = project_root / rel_path
    if not abs_path.exists():
        print(f"File {rel_path} does not exist. Skipping.")
        continue
    
    print(f"Fixing {rel_path}...")
    try:
        # Read the content (which is image data)
        with open(abs_path, "rb") as f:
            img_data = f.read()
        
        # Convert to PDF
        pdf_bytes = img2pdf.convert(img_data)
        
        # Write back as real PDF
        with open(abs_path, "wb") as f:
            f.write(pdf_bytes)
            
        print(f"  ✅ Converted image content to PDF successfully.")
    except Exception as e:
        print(f"  ❌ Error fixing {rel_path}: {e}")
