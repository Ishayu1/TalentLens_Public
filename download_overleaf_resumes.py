import re
from pathlib import Path
from playwright.sync_api import sync_playwright

FAILURE_LOG = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public/manually_need_to_add.txt")

def log_failure(name, link, reason):
    with open(FAILURE_LOG, 'a') as f:
        f.write(f"Name/Index: {name}\nLink: {link}\nReason: {reason}\n{'-'*30}\n")

def download_overleaf_pdf(name, url, output_path, p):
    print(f"  Attempting to download overleaf: {url}")
    try:
        browser = p.chromium.launch(headless=True)
        # Note: Must create context with accept_downloads=True to allow downloading
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(url)
        
        page.wait_for_load_state('domcontentloaded')

        print("  Searching for PDF...")
        selector = 'a[href*="/output.pdf"], a[href*="/download/project"]'
        
        try:
            # wait for it to be attached to the DOM (doesn't have to be visible)
            page.wait_for_selector(selector, state='attached', timeout=15000)
            
            links = page.query_selector_all(selector)
            for download_link in links:
                href = download_link.get_attribute('href')
                if "output.pdf" in href:
                    full_url = "https://www.overleaf.com" + href
                    
                    with page.expect_download() as download_info:
                        page.evaluate(f"window.location.href='{full_url}'")
                    
                    download = download_info.value
                    download.save_as(str(output_path))
                    print(f"  ✅ Saved Overleaf PDF to {output_path}")
                    browser.close()
                    return True
        except Exception as e:
            print(f"  ❌ Failed to find or download via selector: {e}")
            log_failure(name, url, f"Overleaf: {e}")
            
        print("  ❌ Could not find PDF download link on Overleaf page.")
        log_failure(name, url, "Overleaf: PDF link not found or restricted")
        browser.close()
        return False
    except Exception as e:
        print(f"  ❌ Error processing Overleaf link: {str(e)}")
        log_failure(name, url, f"Overleaf Error: {e}")
        if 'browser' in locals():
            browser.close()
        return False

def process_overleaf_placeholders(data_dir, p):
    # Scan for .txt files in ds3 board and member resumes
    search_dirs = [
        data_dir / 'ds3' / 'board_resumes',
        data_dir / 'ds3' / 'member_resumes'
    ]
    
    for s_dir in search_dirs:
        if not s_dir.exists(): continue
        print(f"\nScanning {s_dir.name} for Overleaf links...")
        
        for txt_file in s_dir.glob("*.txt"):
            if txt_file.name == 'board.txt' or txt_file.name == 'member.txt': continue
            
            with open(txt_file, 'r') as f:
                content = f.read()
                
            match = re.search(r'https?://(?:www\.)?overleaf\.com/[^\s]+', content)
            if match:
                overleaf_url = match.group(0)
                pdf_dest = txt_file.with_suffix(".pdf")
                
                # If project link, might be restricted
                if "/project/" in overleaf_url:
                    print(f"  ⚠️ Warning: {overleaf_url} is a project link, likely restricted.")
                
                if download_overleaf_pdf(txt_file.name, overleaf_url, pdf_dest, p):
                    txt_file.unlink()
                    print(f"  ✅ Replaced {txt_file.name} with Overleaf PDF")
                else:
                    # Remove placeholder even on failure so it doesn't clutter the test folder
                    txt_file.unlink()
                    print(f"  ❌ Removed failed Overleaf placeholder {txt_file.name}")

if __name__ == "__main__":
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    print("Starting Overleaf resume scan...")
    with sync_playwright() as p:
        process_overleaf_placeholders(data_dir, p)
    print("\n✓ Overleaf capture process finished.")
