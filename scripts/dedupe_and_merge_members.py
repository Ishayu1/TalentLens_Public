import os
import shutil
import hashlib
import csv
from pathlib import Path

def get_hash(path):
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"Error hashing {path}: {e}")
        return None

def main():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    test_board = base_dir / "test" / "board"
    test_members = base_dir / "test" / "members"
    data_members = base_dir / "data" / "ds3" / "member_resumes"
    
    all_files = []
    
    for dir_path in [test_board, test_members, data_members]:
        if dir_path.exists():
            for f in dir_path.iterdir():
                if f.is_file() and not f.name.startswith('.'):
                    all_files.append(f)
                    
    # Sort files by modification time (newest first). 
    # If duplicates exist, we keep the newest one and remove the older ones.
    all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    seen_hashes = {}
    duplicates = []
    unique_files = []
    
    for f in all_files:
        h = get_hash(f)
        if not h:
            continue
            
        if h in seen_hashes:
            duplicates.append(f)
        else:
            seen_hashes[h] = f
            unique_files.append(f)
            
    print(f"Total files pre-dedupe: {len(all_files)}")
    print(f"Unique files: {len(unique_files)}")
    print(f"Duplicates found: {len(duplicates)}")
    
    # Log duplicates to failed extraction report
    report_path = base_dir / "failed_extractions_report.csv"
    if duplicates:
        with open(report_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for dup in duplicates:
                name = dup.stem.replace('member_resume_', '').replace('board_resume_', '').replace('_', ' ').title()
                writer.writerow([name, str(dup), "Duplicate file (MD5 match found)"])
                
    test_members.mkdir(parents=True, exist_ok=True)
    
    for f in unique_files:
        new_name = f.name
        # Rename board resumes to member resumes as requested
        if new_name.startswith('board_resume_'):
            new_name = new_name.replace('board_resume_', 'member_resume_')
            
        dest_path = test_members / new_name
        
        # Handle filename collisions that are NOT md5 duplicates (e.g. same name but different content)
        if dest_path.exists() and dest_path != f:
            counter = 1
            while (test_members / f"{dest_path.stem}_{counter}{dest_path.suffix}").exists():
                counter += 1
            dest_path = test_members / f"{dest_path.stem}_{counter}{dest_path.suffix}"
            
        if f != dest_path:
            shutil.copy2(str(f), str(dest_path))
            
    # Cleanup originals that were moved or were duplicates
    for f in all_files:
        if f.exists() and f.parent != test_members:
            # We copied unique files to test_members. Now we can safely remove them from test_board and data_members
            f.unlink()
            
    for dup in duplicates:
        if dup.exists():
            dup.unlink()
            
    # Remove old dirs if empty
    for dir_path in [test_board, data_members]:
        if dir_path.exists():
            try:
                # remove directory if empty
                next(dir_path.iterdir())
            except StopIteration:
                shutil.rmtree(dir_path)

    print("Successfully merged and deduplicated board & member resumes into test/members/.")

if __name__ == "__main__":
    main()
