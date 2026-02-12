import os

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Set, List



def get_file_hash(file_path):
    """
    Hash the bit-for-bit content of a file.
    Any files with identical content will be marked as duplicates,
    regardless of their category or filename.
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error hashing content of {file_path}: {e}")
        return hashlib.md5(str(file_path).encode()).hexdigest()

def find_all_resumes(data_dir):
    """Find all resume files in the data directory."""
    resumes = []
    source_folders = [
        'discord/images',
        'discord/pdfs',
        'ds3/board_resumes',
        'ds3/member_resumes',
        'reddit/resumes',
        'resume-dataset/data',
        'Resumes PDF'
    ]
    valid_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.doc', '.docx', '.txt'}
    
    for folder in source_folders:
        folder_path = Path(data_dir) / folder
        
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist, skipping...")
            continue
        
        # Walk through the folder and find all resume files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Skip hidden files and meta files
                if file.startswith('.') or file in ['board.txt', 'member.txt']:
                    continue
                
                file_path = Path(root) / file
                file_ext = file_path.suffix.lower()
                
                if file_ext in valid_extensions:
                    # Determine the source category
                    source = None
                    if 'ds3/board_resumes' in str(file_path):
                        source = 'ds3_board'
                    elif 'ds3/member_resumes' in str(file_path):
                        source = 'ds3_member'
                    elif 'discord' in str(file_path):
                        source = 'discord'
                    elif 'reddit' in str(file_path):
                        source = 'reddit'
                    elif 'resume-dataset' in str(file_path):
                        source = 'resume-dataset'
                    elif 'Resumes PDF' in str(file_path):
                        source = 'resumes_pdf'
                    
                    resumes.append({
                        'path': str(file_path),
                        'filename': file,
                        'source': source,
                        'hash': get_file_hash(file_path),
                        'mtime': os.path.getmtime(file_path)
                    })
    
    return resumes

def check_duplicate(new_file_path, manifest_path=None):
    """
    Check if a new file is a duplicate based on the hash manifest.
    
    Args:
        new_file_path: Path to the new file to check
        manifest_path: Path to the resume_hash_manifest.json file (optional)
                      If None, looks for resume_hash_manifest.json in the script's directory
    
    Returns:
        dict with 'is_duplicate', 'hash', and 'duplicate_path' (path to existing file)
    """
    if manifest_path is None:
        # Use the script's directory to find the manifest
        script_dir = Path(__file__).parent
        manifest_path = script_dir / 'resume_hash_manifest.json'
    
    # Load the manifest
    if not Path(manifest_path).exists():
        return {
            'is_duplicate': False,
            'hash': get_file_hash(new_file_path),
            'duplicate_path': None,
            'error': 'Manifest file not found'
        }
    
    with open(manifest_path, 'r') as f:
        hash_manifest = json.load(f)
    
    # Get hash of new file
    new_hash = get_file_hash(new_file_path)
    
    # Check if hash exists in manifest
    if new_hash in hash_manifest:
        match = hash_manifest[new_hash]
        # Get the actual file path in train/test/val
        base_dir = Path(manifest_path).parent
        duplicate_file_path = base_dir / match['location'] / match['filename']
        
        return {
            'is_duplicate': True,
            'hash': new_hash,
            'duplicate_path': str(duplicate_file_path),
            'location': match['location'],
            'filename': match['filename'],
            'source': match['source']
        }
    else:
        return {
            'is_duplicate': False,
            'hash': new_hash,
            'duplicate_path': None
        }

def get_unique_path(directory, filename):
    """Returns a unique filename by appending a counter if the file exists."""
    path = directory / filename
    if not path.exists():
        return path
    
    counter = 1
    stem = path.stem
    suffix = path.suffix
    while (directory / f"{stem}_{counter}{suffix}").exists():
        counter += 1
    return directory / f"{stem}_{counter}{suffix}"

def organize_resumes(data_dir, train_ratio=0.8):
    """
    Organize resumes into train, test, val, and duplicates folders.
    
    Rules:
    - test: ALL ds3 files (board_resumes + member_resumes)
    - train/val: Everything else, split by train_ratio
    - Duplicates: For each hash, keep the LATEST file (by mtime) in train/test/val.
      Move ALL other occurrences to the duplicates/ folder.
    
    Args:
        data_dir: Path to the data directory
        train_ratio: Ratio of non-ds3 files to put in train vs val (default 0.8)
    """
    # Create output directories
    output_dir = Path(data_dir).parent
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    val_dir = output_dir / 'val'
    duplicates_dir = output_dir / 'duplicates'
    
    # Remove old folders if they exist
    for dir_path in [train_dir, test_dir, val_dir, duplicates_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Find all resumes
    print("\nScanning for resumes...")
    all_resumes = find_all_resumes(data_dir)
    print(f"Found {len(all_resumes)} resume files")
    
    # Track hashes
    hash_to_files = {}
    for resume in all_resumes:
        hash_val = resume['hash']
        if hash_val not in hash_to_files:
            hash_to_files[hash_val] = []
        hash_to_files[hash_val].append(resume)
    
    # Statistics
    stats = {
        'train': 0,
        'test': 0,
        'val': 0,
        'duplicates': 0,
        'total': len(all_resumes)
    }
    
    # Store hash manifest
    hash_manifest = {}
    
    # Track which hashes go where
    ds3_hash_groups = {}
    non_ds3_hash_groups = {}
    
    for hash_val, files in hash_to_files.items():
        # Sort files by mtime (latest first)
        files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Latest one is the "active" file
        active_file = files[0]
        duplicates = files[1:]
        
        if active_file['source'] in ['ds3_board', 'ds3_member']:
            ds3_hash_groups[hash_val] = {'active': active_file, 'duplicates': duplicates}
        else:
            non_ds3_hash_groups[hash_val] = {'active': active_file, 'duplicates': duplicates}

    # Process ds3 -> test
    print(f"\nProcessing {len(ds3_hash_groups)} ds3 hash groups -> test/")
    for hash_val, group in ds3_hash_groups.items():
        active = group['active']
        dest_path = get_unique_path(test_dir, active['filename'])
        shutil.copy2(active['path'], dest_path)
        stats['test'] += 1
        
        # Manifest entry for active
        hash_manifest[hash_val] = {
            'filename': dest_path.name,
            'original_path': active['path'],
            'source': active['source'],
            'location': 'test',
            'duplicates': []
        }
        
        # Process duplicates
        for dup in group['duplicates']:
            dup_dest = get_unique_path(duplicates_dir, f"{dup['source']}_{dup['filename']}")
            shutil.copy2(dup['path'], dup_dest)
            stats['duplicates'] += 1
            hash_manifest[hash_val]['duplicates'].append({
                'filename': dup_dest.name,
                'original_path': dup['path'],
                'source': dup['source'],
                'duplicate_of': f"test/{dest_path.name}"
            })
    
    # Split non-ds3 between train and val
    import random
    random.seed(42)
    hash_list = list(non_ds3_hash_groups.keys())
    random.shuffle(hash_list)
    
    train_split_index = int(len(hash_list) * train_ratio)
    
    print(f"\nProcessing {len(hash_list)} non-ds3 hash groups:")
    print(f"  - {train_split_index} -> train/")
    print(f"  - {len(hash_list) - train_split_index} -> val/")
    
    for i, hash_val in enumerate(hash_list):
        group = non_ds3_hash_groups[hash_val]
        active = group['active']
        
        if i < train_split_index:
            dest_dir = train_dir
            location = 'train'
            stats['train'] += 1
        else:
            dest_dir = val_dir
            location = 'val'
            stats['val'] += 1
            
        dest_path = get_unique_path(dest_dir, active['filename'])
        shutil.copy2(active['path'], dest_path)
        
        # Manifest entry for active
        hash_manifest[hash_val] = {
            'filename': dest_path.name,
            'original_path': active['path'],
            'source': active['source'],
            'location': location,
            'duplicates': []
        }
        
        # Process duplicates
        for dup in group['duplicates']:
            dup_dest = get_unique_path(duplicates_dir, f"{dup['source']}_{dup['filename']}")
            shutil.copy2(dup['path'], dup_dest)
            stats['duplicates'] += 1
            hash_manifest[hash_val]['duplicates'].append({
                'filename': dup_dest.name,
                'original_path': dup['path'],
                'source': dup['source'],
                'duplicate_of': f"{location}/{dest_path.name}"
            })
            
    # Save hash manifest
    manifest_path = output_dir / 'resume_hash_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(hash_manifest, f, indent=2)
    
    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE")
    print("="*60)
    print(f"Total files found: {stats['total']}")
    print(f"  - Train: {stats['train']}")
    print(f"  - Val: {stats['val']}")
    print(f"  - Test: {stats['test']} (all ds3 files)")
    print(f"  - Duplicates: {stats['duplicates']}")
    print(f"\nHash manifest saved to: {manifest_path}")
    print(f"Total unique resumes: {len(hash_manifest)}")

if __name__ == "__main__":
    # Set the data directory path relative to script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    print("Starting resume organization...")
    print(f"Data directory: {data_dir}")
    print("\nRules:")
    print("  - test: ALL ds3 files")
    print("  - train/val: Everything else (80/20 split)")
    print("  - Duplicates: Move older versions to duplicates/ folder\n")
    
    organize_resumes(str(data_dir), train_ratio=0.8)
    
    print("\n✓ Done! You can now use check_duplicate() function to verify new uploads.")
