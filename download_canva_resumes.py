import re
from pathlib import Path
from playwright.sync_api import sync_playwright

FAILURE_LOG = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public/manually_need_to_add.txt")

def log_failure(name, link, reason):
    with open(FAILURE_LOG, 'a') as f:
        f.write(f"Name/Index: {name}\nLink: {link}\nReason: {reason}\n{'-'*30}\n")

def capture_web_pdf(name, url, output_path, p):
    print(f"  Attempting to capture: {url}")
    try:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 1600})
        page = context.new_page()
        
        page.goto(url)
        
        try:
            page.wait_for_load_state('networkidle', timeout=30000)
        except Exception:
            pass
            
        page.wait_for_timeout(5000)
        if not page.title():
             log_failure(name, url, "Page has no title (failed to load)")
             browser.close()
             return False

        page.pdf(path=str(output_path), format="A4", print_background=True)
        
        print(f"  ✅ Saved PDF (via print-to-pdf) to {output_path}")
        browser.close()
        return True
    except Exception as e:
        print(f"  ❌ Error processing web link: {str(e)}")
        log_failure(name, url, f"Web capture failed: {e}")
        if 'browser' in locals():
            browser.close()
        return False

def process_web_placeholders(data_dir, p):
    # Scan for .txt files in ds3 board and member resumes
    search_dirs = [
        data_dir / 'ds3' / 'board_resumes',
        data_dir / 'ds3' / 'member_resumes'
    ]
    
    for s_dir in search_dirs:
        if not s_dir.exists(): continue
        print(f"\nScanning {s_dir.name} for Canva/pdflink links...")
        
        for txt_file in s_dir.glob("*.txt"):
            if txt_file.name == 'board.txt' or txt_file.name == 'member.txt': continue
            
            with open(txt_file, 'r') as f:
                content = f.read()
                
            match = re.search(r'https?://(?:www\.)?(?:canva\.com|pdflink\.to)/[^\s]+', content)
            if match:
                target_url = match.group(0)
                pdf_dest = txt_file.with_suffix(".pdf")
                
                if capture_web_pdf(txt_file.name, target_url, pdf_dest, p):
                    txt_file.unlink()
                    print(f"  ✅ Replaced {txt_file.name} with PDF")
                else:
                    # Remove placeholder even on failure so it doesn't clutter the test folder
                    txt_file.unlink()
                    print(f"  ❌ Removed failed web placeholder {txt_file.name}")

if __name__ == "__main__":
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    print("Starting Web (Canva/pdflink) resume capture...")
    with sync_playwright() as p:
        process_web_placeholders(data_dir, p)
    print("\n✓ Web capture process finished.")
