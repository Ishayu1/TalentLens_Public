import os
import re
import requests
import json
from pathlib import Path

def get_google_drive_download_url(url):
    file_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if file_match:
        file_id = file_match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    
    # Improved Google Doc export logic with tab support
    doc_match = re.search(r'docs\.google\.com/document/d/([a-zA-Z0-9_-]+)', url)
    if doc_match:
        doc_id = doc_match.group(1)
        export_url = f'https://docs.google.com/document/d/{doc_id}/export?format=pdf'
        
        # Check for tab parameter (e.g., ?tab=t.l6yjdwqyotkg)
        tab_match = re.search(r'[?&]tab=([^&]+)', url)
        if tab_match:
            tab_id = tab_match.group(1)
            export_url += f'&tab={tab_id}'
            
        return export_url
    
    return None

def download_file(url, output_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()    
        
        # Check if we got HTML instead of a PDF
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Peek at the first chunk to verify the file type
        # We use a generator to get the first chunk and then continue
        chunks = response.iter_content(chunk_size=8192)
        try:
            first_chunk = next(chunks)
        except StopIteration:
            return False

        if 'text/html' in content_type or first_chunk.startswith(b'<') or b'<html' in first_chunk.lower():
            # If it's HTML, check if it's REALLY a PDF (misreported content type)
            if not first_chunk.startswith(b"%PDF-"):
                print(f"  ❌ Error: Received HTML instead of PDF for {url}")
                return False
        
        # Check if it's an image that needs conversion
        is_image = 'image/' in content_type or first_chunk.startswith(b'\x89PNG') or first_chunk.startswith(b'\xff\xd8')
        
        if is_image:
            print(f"  📸 Detected image for {url}, converting to PDF...")
            import img2pdf
            # Need regular list for img2pdf
            all_data = first_chunk + b"".join(chunks)
            try:
                pdf_data = img2pdf.convert(all_data)
                with open(output_path, 'wb') as f:
                    f.write(pdf_data)
                return True
            except Exception as e:
                print(f"  ❌ Image conversion failed: {e}")
                return False

        with open(output_path, 'wb') as f:
            f.write(first_chunk)
            for chunk in chunks:
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ❌ Error downloading {url}: {e}")
        return False

FAILURE_LOG = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public/manually_need_to_add.txt")

def log_failure(name, link, reason):
    with open(FAILURE_LOG, 'a') as f:
        f.write(f"Name/Index: {name}\nLink: {link}\nReason: {reason}\n{'-'*30}\n")

def process_board_links(board_txt, output_dir):
    print(f"\nProcessing board links from {board_txt}...")
    if not board_txt.exists():
        print(f"Error: {board_txt} not found.")
        return

    # CLEANUP: Remove old board resumes
    print("  Removing old board resumes...")
    for f in output_dir.glob("board_resume_*.pdf"): f.unlink()
    for f in output_dir.glob("board_resume_*.txt"): f.unlink()

    with open(board_txt, 'r') as f:
        links = [line.strip() for line in f if line.strip()]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, link in enumerate(links):
        index_name = f"board_resume_{i+1}"
        
        if not link or link == "-" or link.lower() == "none" or link.lower() == "n/a":
            continue

        if 'drive.google.com' in link or 'docs.google.com' in link:
            download_url = get_google_drive_download_url(link)
            if download_url:
                filename = f"{index_name}.pdf"
                dest = output_dir / filename
                print(f"  Downloading GDoc/Drive: {index_name}...")
                if download_file(download_url, dest):
                    print(f"  ✅ Saved PDF to {dest}")
                else:
                    log_failure(index_name, link, "cannot add need permission")
            else:
                log_failure(index_name, link, "cannot add need permission")
        
        elif 'overleaf.com' in link:
            # Overleaf handled by dedicated script; placeholder here
            filename = f"{index_name}.txt"
            dest = output_dir / filename
            with open(dest, 'w') as f_out:
                f_out.write(f"Link: {link}\n")
            print(f"  📝 Saved Overleaf placeholder to {dest}")

        elif 'canva.com' in link or 'pdflink.to' in link:
            filename = f"{index_name}.txt"
            dest = output_dir / filename
            with open(dest, 'w') as f_out:
                f_out.write(f"Link: {link}\n")
            print(f"  📝 Saved Canva/pdflink placeholder to {dest}")

        elif link.lower().endswith('.pdf'):
            filename = f"{index_name}.pdf"
            dest = output_dir / filename
            if download_file(link, dest):
                print(f"  ✅ Saved PDF to {dest}")
            else:
                log_failure(index_name, link, "alternate link (HTML returned or failed)")
        else:
            log_failure(index_name, link, "alternate link")

def process_member_csv(csv_path, output_dir):
    import csv
    print(f"\nProcessing member links from {csv_path}...")
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return

    # CLEANUP: Remove old member resumes
    print("  Removing old member resumes...")
    for f in output_dir.glob("member_resume_*.pdf"): f.unlink()
    for f in output_dir.glob("member_resume_*.txt"): f.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_name = row.get('Full Name', 'unknown').strip()
                link = row.get('Resume Link', '').strip()
                if not full_name:
                    continue
                
                if not link or link == "-" or link.lower() == "none" or link.lower() == "n/a":
                    continue
                    
                name_slug = full_name.lower().replace(' ', '_')
                print(f"  Processing {full_name}...")
                
                if 'drive.google.com' in link or 'docs.google.com' in link:
                    download_url = get_google_drive_download_url(link)
                    if download_url:
                        filename = f"member_resume_{name_slug}.pdf"
                        dest = output_dir / filename
                        if download_file(download_url, dest):
                            print(f"  ✅ Saved to {dest}")
                        else:
                            log_failure(full_name, link, "cannot add need permission")
                    else:
                        log_failure(full_name, link, "cannot add need permission")
                            
                elif 'overleaf.com' in link:
                    filename = f"member_resume_{name_slug}.txt"
                    dest = output_dir / filename
                    with open(dest, 'w') as f_out:
                        f_out.write(f"Name: {full_name}\n")
                        f_out.write(f"Link: {link}\n")
                    print(f"  📝 Saved Overleaf placeholder to {dest}")

                elif 'canva.com' in link or 'pdflink.to' in link:
                    filename = f"member_resume_{name_slug}.txt"
                    dest = output_dir / filename
                    with open(dest, 'w') as f_out:
                        f_out.write(f"Name: {full_name}\n")
                        f_out.write(f"Link: {link}\n")
                    print(f"  📝 Saved Canva/pdflink placeholder to {dest}")

                elif link.lower().endswith('.pdf'):
                    filename = f"member_resume_{name_slug}.pdf"
                    dest = output_dir / filename
                    if download_file(link, dest):
                        print(f"  ✅ Saved PDF to {dest}")
                    else:
                        log_failure(full_name, link, "alternate link (HTML returned or failed)")
                else:
                    log_failure(full_name, link, "alternate link")

    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    # Clear old failure log
    if FAILURE_LOG.exists(): FAILURE_LOG.unlink()
    
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public/data/ds3")
    
    # Process Board Resumes
    board_txt = base_dir / "board_resumes" / "board.txt"
    board_output = base_dir / "board_resumes" 
    process_board_links(board_txt, board_output)
    
    # Process Member Resumes 
    member_csv = base_dir / "member_resumes" / "members.csv"
    member_output = base_dir / "member_resumes"
    process_member_csv(member_csv, member_output)
    
    print("\n✓ Download process finished.")
