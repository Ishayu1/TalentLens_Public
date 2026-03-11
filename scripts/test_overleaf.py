from playwright.sync_api import sync_playwright
import time

def test_download():
    url = "https://www.overleaf.com/read/wkmsvmqtyszp"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(url)
        try:
            page.wait_for_load_state('networkidle', timeout=15000)
        except Exception:
            pass
        
        print("Page loaded. Looking for download PDF anchor tags...")
        try:
            links = page.query_selector_all('a[href*="/output.pdf"], a[href*="/download/project"]')
            for l in links:
                href = l.get_attribute('href')
                print("Found link:", href)
                if "output.pdf" in href:
                    print("Attempting to download by navigating...")
                    full_url = "https://www.overleaf.com" + href
                    
                    with page.expect_download(timeout=10000) as download_info:
                        page.evaluate(f"window.location.href='{full_url}'")
                    
                    download = download_info.value
                    download.save_as("test_output.pdf")
                    print("Successfully downloaded PDF via 'Download PDF' link navigation.")
                    return
            print("No suitable PDF link found.")
        except Exception as e:
            print("Failed:", e)

if __name__ == "__main__":
    test_download()
