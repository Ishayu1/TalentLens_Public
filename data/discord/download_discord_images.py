#!/usr/bin/env python3
"""
Script to download files (images and PDFs) from .webloc and .url files in data/discord/links/
"""

import configparser
import hashlib
import os
import plistlib
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def extract_url_from_webloc(webloc_path):
    """Extract URL from a macOS .webloc file (plist format)"""
    try:
        with open(webloc_path, 'rb') as f:
            plist_data = plistlib.load(f)
        return plist_data.get('URL')
    except Exception as e:
        print(f"✗ Error reading {webloc_path.name}: {e}")
        return None


def extract_url_from_url_file(url_path):
    """Extract URL from a Windows .url file (INI format)"""
    try:
        config = configparser.ConfigParser()
        config.read(url_path, encoding='utf-8')
        return config.get('InternetShortcut', 'URL', fallback=None)
    except Exception as e:
        print(f"✗ Error reading {url_path.name}: {e}")
        return None


def extract_url(link_path):
    """Extract URL from either a .webloc or .url file"""
    if link_path.suffix.lower() == '.webloc':
        return extract_url_from_webloc(link_path)
    elif link_path.suffix.lower() == '.url':
        return extract_url_from_url_file(link_path)
    return None


KNOWN_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'}


def get_filename_from_link(link_path, url):
    """Get a clean filename from the .webloc/.url filename or URL"""
    # Remove the link extension (.webloc or .url)
    webloc_name = link_path.stem

    # Check if the webloc filename has a known file extension (like .pdf or .png)
    ext = os.path.splitext(webloc_name)[1].lower()
    if ext in KNOWN_EXTENSIONS:
        return webloc_name

    # Otherwise, extract the real filename from the URL
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    # Clean query params
    if '?' in filename:
        filename = filename.split('?')[0]

    # If URL filename is generic (like "image.png"), prefix with webloc index
    # to avoid collisions
    if filename in ('image.png', 'image.jpg', 'image.jpeg'):
        # Extract a number from the webloc name if possible
        parts = webloc_name.split()
        suffix = parts[-1] if parts else '0'
        name, url_ext = os.path.splitext(filename)
        filename = f"image_{suffix}{url_ext}"

    return filename


def get_file_type(filename):
    """Determine if file is PDF or image based on extension"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        return 'image'
    return 'other'


def hash_file(filepath):
    """Compute SHA-256 hash of a file's contents"""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, output_path):
    """Download file from URL to output path"""
    # Download with proper headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        with open(output_path, 'wb') as out_file:
            out_file.write(response.read())

    return output_path


def main():
    # Define paths
    base_dir = Path(__file__).parent
    links_dir = base_dir / "links"
    base_output_dir = base_dir

    # Create subdirectories for organization
    images_dir = base_output_dir / "images"
    pdfs_dir = base_output_dir / "pdfs"
    images_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Find all .webloc and .url files
    link_files = sorted(
        list(links_dir.glob("*.webloc")) + list(links_dir.glob("*.url"))
    )

    if not link_files:
        print("No .webloc or .url files found in data/discord/links/")
        return

    print(f"Found {len(link_files)} link file(s)\n")

    # Build hash index of already-downloaded files for content dedup
    seen_hashes = {}  # hash -> filename
    print("Building hash index of existing files...")
    for d in [images_dir, pdfs_dir]:
        for f in d.iterdir():
            if f.is_file():
                h = hash_file(f)
                seen_hashes[h] = f.name
    print(f"Indexed {len(seen_hashes)} existing file(s)\n")

    # Track statistics
    stats = {
        'total': len(link_files),
        'downloaded': 0,
        'skipped_filename': 0,
        'skipped_duplicate': 0,
        'failed': 0
    }
    duplicates = []  # list of (new_filename, existing_filename)

    # Process each link file
    for idx, link_path in enumerate(link_files, 1):
        print(f"[{idx}/{stats['total']}] Processing: {link_path.name}")

        try:
            # Extract URL from .webloc or .url file
            url = extract_url(link_path)

            if not url:
                print(f"✗ Could not extract URL")
                stats['failed'] += 1
                print()
                continue

            # Get filename and determine file type
            filename = get_filename_from_link(link_path, url)
            file_type = get_file_type(filename)

            # Determine output directory based on file type
            if file_type == 'pdf':
                output_dir = pdfs_dir
            elif file_type == 'image':
                output_dir = images_dir
            else:
                output_dir = base_output_dir

            output_path = output_dir / filename

            # Skip if file with same name already downloaded
            if output_path.exists():
                file_size = output_path.stat().st_size
                print(f"⊙ Already exists ({file_size:,} bytes): {filename}")
                stats['skipped_filename'] += 1
                print()
                continue

            # Download the file
            print(f"↓ Downloading: {filename}")
            download_file(url, output_path)

            # Check for content duplicate
            file_hash = hash_file(output_path)
            if file_hash in seen_hashes:
                existing = seen_hashes[file_hash]
                print(f"⊘ Duplicate content! Same as: {existing}")
                duplicates.append((filename, existing))
                output_path.unlink()  # remove the duplicate
                stats['skipped_duplicate'] += 1
            else:
                file_size = output_path.stat().st_size
                print(f"✓ Downloaded ({file_size:,} bytes): {filename}")
                seen_hashes[file_hash] = filename
                stats['downloaded'] += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            stats['failed'] += 1

        print()

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:        {stats['total']}")
    print(f"Downloaded:         {stats['downloaded']}")
    print(f"Already existed:    {stats['skipped_filename']}")
    print(f"Content duplicates: {stats['skipped_duplicate']}")
    print(f"Failed:             {stats['failed']}")

    if duplicates:
        print(f"\nDuplicate files removed:")
        for new_name, existing_name in duplicates:
            print(f"  {new_name} == {existing_name}")

    print(f"\nFiles saved to:")
    print(f"  Images: {images_dir}")
    print(f"  PDFs:   {pdfs_dir}")


if __name__ == "__main__":
    main()
