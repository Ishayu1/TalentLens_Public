import shutil
from pathlib import Path

def finalize_manual():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    manual_dir = base_dir / "manually extracted"
    test_members = base_dir / "test" / "members"
    
    mapping = {
        "My PDF.pdf": "anh_long_do",
        "Amy Nguyen Trinh - Resume (1).pdf": "amy_trinh",
        "My Resume - Ayse Sule Ekiz.pdf": "ayse_ekiz",
        "RAMBLE.pdf": "charlotte_mundo"
    }

    for orig_name, slug in mapping.items():
        orig_path = manual_dir / orig_name
        if orig_path.exists():
            target_path = test_members / f"member_resume_{slug}.pdf"
            
            # Delete target if it exists to overwrite with manual extraction
            if target_path.exists():
                print(f"Overwriting {target_path.name}")
                target_path.unlink()
                
            # If text placeholder exists, remove it
            txt_path = test_members / f"member_resume_{slug}.txt"
            if txt_path.exists():
                txt_path.unlink()
                
            shutil.move(str(orig_path), str(target_path))
            print(f"Mapped {orig_name} to {slug}")

    # Remove the manually extracted folder since it's empty
    try:
        manual_dir.rmdir()
        print("Removed 'manually extracted' folder.")
    except Exception as e:
        print("Could not remove 'manually extracted' folder (might not be empty):", e)

if __name__ == "__main__":
    finalize_manual()
