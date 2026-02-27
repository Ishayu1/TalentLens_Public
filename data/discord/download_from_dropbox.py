#!/usr/bin/env python3
"""
Download discord resume images and PDFs from a shared Dropbox folder link.

No API token or Dropbox account required — uses the public shared link directly.

Usage:
    python download_from_dropbox.py <dropbox-shared-link>

Example:
    python download_from_dropbox.py "https://www.dropbox.com/scl/fo/jepsuisabfchr3mn70v3v/AJ6PRx4-b0IURZ_Lb_Ox56E?rlkey=uqsahacirzq5noxhlrmbeefdp&st=za2smdwe&dl=0"

This will download and extract:
    - images/ folder -> data/discord/images/
    - pdfs/   folder -> data/discord/pdfs/
"""

import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urlencode, parse_qs


def convert_to_direct_download(shared_url):
    """Convert a Dropbox shared folder URL to a direct download (zip) URL."""
    parsed = urlparse(shared_url)
    params = parse_qs(parsed.query)

    # Set dl=1 to force direct download as zip
    params["dl"] = ["1"]

    new_query = urlencode({k: v[0] for k, v in params.items()})
    direct_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"
    return direct_url


def download_zip(url):
    """Download the zip file from Dropbox into memory and return the bytes."""
    print(f"Downloading zip from Dropbox...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        data = response.read()
    print(f"Downloaded {len(data):,} bytes")
    return data


def extract_and_place(zip_data, base_dir):
    """Extract the zip and place images/ and pdfs/ into the right local folders."""
    images_dir = base_dir / "images"
    pdfs_dir = base_dir / "pdfs"
    images_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "pdfs": 0, "skipped": 0, "other": 0}

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        print(f"Zip contains {len(members)} file(s)\n")

        for member in sorted(members):
            # Dropbox zips have a top-level folder name, e.g.:
            #   "FolderName/images/file.png"
            #   "FolderName/pdfs/file.pdf"
            parts = Path(member).parts
            filename = parts[-1]

            # Skip hidden/system files
            if filename.startswith(".") or filename == "__MACOSX":
                continue

            # Find which subfolder this belongs to by checking path components
            member_lower = member.lower()

            if "/images/" in member_lower or "\\images\\" in member_lower:
                dest = images_dir / filename
                category = "images"
            elif "/pdfs/" in member_lower or "\\pdfs\\" in member_lower:
                dest = pdfs_dir / filename
                category = "pdfs"
            else:
                # File is at root level or unknown subfolder — guess by extension
                ext = os.path.splitext(filename)[1].lower()
                if ext == ".pdf":
                    dest = pdfs_dir / filename
                    category = "pdfs"
                elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}:
                    dest = images_dir / filename
                    category = "images"
                else:
                    print(f"  [skip] Unknown file type: {member}")
                    stats["other"] += 1
                    continue

            # Skip if already exists with same size
            file_data = zf.read(member)
            if dest.exists() and dest.stat().st_size == len(file_data):
                print(f"  [skip] {category}/{filename} (already exists, same size)")
                stats["skipped"] += 1
                continue

            dest.write_bytes(file_data)
            print(f"  [saved] {category}/{filename} ({len(file_data):,} bytes)")
            stats[category] += 1

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_from_dropbox.py <dropbox-shared-link>")
        print()
        print("Example:")
        print('  python download_from_dropbox.py "https://www.dropbox.com/scl/fo/..."')
        sys.exit(1)

    shared_url = sys.argv[1]

    # Validate it looks like a Dropbox URL
    if "dropbox.com" not in shared_url:
        print("Error: URL doesn't look like a Dropbox link.")
        sys.exit(1)

    base_dir = Path(__file__).parent

    # Convert to direct download link
    direct_url = convert_to_direct_download(shared_url)

    # Download the zip
    zip_data = download_zip(direct_url)

    # Extract and place files
    print("\nExtracting files...\n")
    stats = extract_and_place(zip_data, base_dir)

    # Summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Images downloaded:  {stats['images']}")
    print(f"PDFs downloaded:    {stats['pdfs']}")
    print(f"Skipped (exists):   {stats['skipped']}")
    print(f"Other/unknown:      {stats['other']}")
    print(f"\nFiles saved to:")
    print(f"  Images: {base_dir / 'images'}")
    print(f"  PDFs:   {base_dir / 'pdfs'}")


if __name__ == "__main__":
    main()
