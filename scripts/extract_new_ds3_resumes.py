import csv
import os
import re
import requests
import string
from pathlib import Path

def get_google_drive_download_url(url):
    file_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if file_match:
        file_id = file_match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    
    doc_match = re.search(r'docs\.google\.com/document/d/([a-zA-Z0-9_-]+)', url)
    if doc_match:
        doc_id = doc_match.group(1)
        export_url = f'https://docs.google.com/document/d/{doc_id}/export?format=pdf'
        
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
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        chunks = response.iter_content(chunk_size=8192)
        try:
            first_chunk = next(chunks)
        except StopIteration:
            return False

        if 'text/html' in content_type or first_chunk.startswith(b'<') or b'<html' in first_chunk.lower():
            if not first_chunk.startswith(b"%PDF-"):
                return False
        
        is_image = 'image/' in content_type or first_chunk.startswith(b'\x89PNG') or first_chunk.startswith(b'\xff\xd8')
        
        if is_image:
            import img2pdf
            all_data = first_chunk + b"".join(chunks)
            try:
                pdf_data = img2pdf.convert(all_data)
                with open(output_path, 'wb') as f:
                    f.write(pdf_data)
                return True
            except Exception:
                return False

        with open(output_path, 'wb') as f:
            f.write(first_chunk)
            for chunk in chunks:
                f.write(chunk)
        return True
    except Exception:
        return False

def clean_filename(name):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned = ''.join(c for c in name if c in valid_chars)
    return cleaned.lower().replace(' ', '_')

def log_failure(csv_writer, full_name, link, reason):
    csv_writer.writerow([full_name, link, reason])
    print(f"  ❌ Failed [{full_name}]: {reason}")

def main():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    csv_path = base_dir / "Supabase Snippet Talent Pool Resume Links.csv"
    output_dir = base_dir / "data" / "ds3" / "member_resumes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    failure_log_path = base_dir / "failed_extractions_report.csv"
    
    print(f"Starting extraction from {csv_path}...")
    
    with open(failure_log_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Full Name', 'Resume Link', 'Reason'])
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                for index, row in enumerate(reader, start=1):
                    full_name = row.get('full_name', '').strip()
                    if not full_name:
                        full_name = f"Unknown_Candidate_{index}"
                        
                    link = row.get('resume_link', '').strip()
                    if not link or link == "-" or link.lower() in ["none", "n/a", "null"]:
                        log_failure(writer, full_name, link, "No valid link provided")
                        continue
                        
                    name_slug = clean_filename(full_name)
                    print(f"[{index}] Processing {full_name}...")
                    
                    if 'drive.google.com' in link or 'docs.google.com' in link:
                        download_url = get_google_drive_download_url(link)
                        if download_url:
                            filename = f"member_resume_{name_slug}.pdf"
                            dest = output_dir / filename
                            if download_file(download_url, dest):
                                print(f"  ✅ Saved to {dest.name}")
                            else:
                                log_failure(writer, full_name, link, "Download failed (Needs permission or generic error)")
                        else:
                            log_failure(writer, full_name, link, "Invalid Google Drive/Docs link format")
                                
                    elif 'overleaf.com' in link:
                        filename = f"member_resume_{name_slug}.txt"
                        dest = output_dir / filename
                        with open(dest, 'w') as fh:
                            fh.write(f"Name: {full_name}\nLink: {link}\n")
                        print(f"  📝 Saved Overleaf placeholder to {dest.name}")

                    elif 'canva.com' in link or 'pdflink.to' in link:
                        filename = f"member_resume_{name_slug}.txt"
                        dest = output_dir / filename
                        with open(dest, 'w') as fh:
                            fh.write(f"Name: {full_name}\nLink: {link}\n")
                        print(f"  📝 Saved Canva/pdflink placeholder to {dest.name}")

                    elif link.lower().endswith('.pdf'):
                        filename = f"member_resume_{name_slug}.pdf"
                        dest = output_dir / filename
                        if download_file(link, dest):
                            print(f"  ✅ Saved PDF to {dest.name}")
                        else:
                            log_failure(writer, full_name, link, "Alternate link (HTML returned or failed)")
                    else:
                        log_failure(writer, full_name, link, "Unsupported link format")

        except Exception as e:
            print(f"Error processing CSV: {e}")

    print(f"\nExtraction complete. Check '{failure_log_path}' for any failures.")

if __name__ == '__main__':
    main()
