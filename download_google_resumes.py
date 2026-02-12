
import os
import re
import json
import requests
from pathlib import Path

def get_google_drive_download_url(url):
    file_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if file_match:
        file_id = file_match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    
    doc_match = re.search(r'docs\.google\.com/document/d/([a-zA-Z0-9_-]+)', url)
    if doc_match:
        doc_id = doc_match.group(1)
        return f'https://docs.google.com/document/d/{doc_id}/export?format=pdf'
    
    return None

def download_file(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()    
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ❌ Error downloading {url}: {e}")
        return False

def process_board_links(txt_path, output_dir):
    print(f"\nProcessing board links from {txt_path}...")
    if not txt_path.exists():
        print(f"Warning: {txt_path} not found.")
        return

    with open(txt_path, 'r') as f:
        links = [line.strip().strip('"') for line in f if line.strip()]

    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, link in enumerate(links):
        if 'drive.google.com' in link or 'docs.google.com' in link:
            download_url = get_google_drive_download_url(link)
            if download_url:
                filename = f"board_resume_{i+1}.pdf"
                dest = output_dir / filename
                if download_file(download_url, dest):
                    print(f"  ✅ Saved to {dest}")
                else:
                    if dest.exists(): dest.unlink() # Remove partial/failed files
        else:
            print(f"  ⚠️ Skipping non-Google link: {link}")

def process_member_json(json_path, output_dir):
    print(f"\nProcessing member links from {json_path}...")
    if not json_path.exists():
        print(f"Warning: {json_path} not found.")
        return

    try:
        with open(json_path, 'r') as f:
            members = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {json_path} is not a valid JSON file.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    for member in members:
        name = member.get('full_name', 'unknown').replace(' ', '_').lower()
        link = member.get('resume_link')
        
        if link and ('drive.google.com' in link or 'docs.google.com' in link):
            download_url = get_google_drive_download_url(link)
            if download_url:
                filename = f"member_resume_{name}.pdf"
                dest = output_dir / filename
                print(f"  Downloading resume for {member.get('full_name')}...")
                if download_file(download_url, dest):
                    print(f"  ✅ Saved to {dest}")
                else:
                    if dest.exists(): dest.unlink()

if __name__ == "__main__":
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public/data/ds3")
    
    # Process Board Resumes
    board_txt = base_dir / "board_resumes" / "board.txt"
    board_output = base_dir / "board_resumes" # Store PDFs in the same place
    process_board_links(board_txt, board_output)
    
    # Process Member Resumes
    member_json = base_dir / "member_resumes" / "member.txt"
    member_output = base_dir / "member_resumes"
    process_member_json(member_json, member_output)
    
    print("\n✓ Download process finished.")
