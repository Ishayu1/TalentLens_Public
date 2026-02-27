#!/usr/bin/env python3
"""
Script to upload discord images and PDFs to the shared Dropbox folder (Talentlens).

Prerequisites:
    pip install dropbox python-dotenv

Setup:
    1. Go to https://www.dropbox.com/developers/apps
    2. Click "Create app"
    3. Choose "Scoped access" -> "Full Dropbox" -> name it (e.g. "TalentLens Uploader")
    4. Under Permissions tab, enable: files.content.write, files.content.read, sharing.write
    5. Under Settings tab, click "Generate" under "Generated access token"
    6. Paste the token in the .env file next to this script:
         DROPBOX_ACCESS_TOKEN=your_token_here
"""

import hashlib
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: 'python-dotenv' package not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    import dropbox
    from dropbox.common import PathRoot
    from dropbox.exceptions import ApiError
    from dropbox.files import WriteMode
except ImportError:
    print("Error: 'dropbox' package not installed. Run: pip install dropbox")
    sys.exit(1)

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")


# Dropbox shared folder URL (for reference)
DROPBOX_SHARED_FOLDER_URL = (
    "https://www.dropbox.com/scl/fo/jepsuisabfchr3mn70v3v/"
    "AJ6PRx4-b0IURZ_Lb_Ox56E?rlkey=uqsahacirzq5noxhlrmbeefdp&st=b4jn5dw1&dl=0"
)

# Path inside the shared team folder
DROPBOX_BASE_PATH = "/Talentlens/Discord_Resumes"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
PDF_EXTENSIONS = {".pdf"}

DROPBOX_HASH_BLOCK_SIZE = 4 * 1024 * 1024  # 4 MB


def dropbox_content_hash(file_path):
    """Compute the Dropbox content hash for a local file.

    Dropbox splits files into 4MB blocks, SHA-256 hashes each block,
    then SHA-256 hashes the concatenated block hashes.
    """
    block_hashes = b""
    with open(file_path, "rb") as f:
        while True:
            block = f.read(DROPBOX_HASH_BLOCK_SIZE)
            if not block:
                break
            block_hashes += hashlib.sha256(block).digest()
    return hashlib.sha256(block_hashes).hexdigest()


def get_dropbox_client():
    """Create and return an authenticated Dropbox client scoped to the team root."""
    token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    if not token:
        print("Error: DROPBOX_ACCESS_TOKEN not set.")
        print("Paste your token in the .env file next to this script.")
        sys.exit(1)

    dbx = dropbox.Dropbox(token)

    # Get the team root namespace so we can access shared folders
    account = dbx.users_get_current_account()
    root_namespace_id = account.root_info.root_namespace_id
    dbx = dbx.with_path_root(PathRoot.root(root_namespace_id))

    return dbx, account.name.display_name


def upload_file(dbx, local_path, dropbox_path, remote_hashes):
    """Upload a single file to Dropbox with content-hash deduplication.

    Checks:
    1. If the same filename already exists with the same content hash -> skip
    2. If a different filename has the same content hash -> skip (duplicate)
    3. Otherwise -> upload
    """
    file_size = local_path.stat().st_size
    local_hash = dropbox_content_hash(local_path)

    # Check if file already exists at the same path with same content
    try:
        metadata = dbx.files_get_metadata(dropbox_path)
        if metadata.content_hash == local_hash:
            print(f"  [skip] Already exists (same content): {local_path.name}")
            return "skipped"
    except ApiError as e:
        if e.error.is_path() and e.error.get_path().is_not_found():
            pass  # File doesn't exist yet
        else:
            raise

    # Check if the same content exists under a different filename on Dropbox
    if local_hash in remote_hashes:
        existing = remote_hashes[local_hash]
        print(f"  [skip] Duplicate content: {local_path.name} == {existing}")
        return "duplicate"

    # Upload the file
    with open(local_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=WriteMode.overwrite)
    print(f"  [uploaded] {dropbox_path} ({file_size:,} bytes)")
    remote_hashes[local_hash] = local_path.name
    return "uploaded"


def ensure_folder_exists(dbx, path):
    """Create a folder on Dropbox if it doesn't exist."""
    try:
        dbx.files_get_metadata(path)
    except ApiError as e:
        if e.error.is_path() and e.error.get_path().is_not_found():
            dbx.files_create_folder_v2(path)
            print(f"  Created folder: {path}")
        else:
            raise


def main():
    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"
    pdfs_dir = base_dir / "pdfs"

    # Verify local directories exist
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        print("Run download_discord_images.py first to download the files.")
        sys.exit(1)

    if not pdfs_dir.exists():
        print(f"Error: PDFs directory not found: {pdfs_dir}")
        print("Run download_discord_images.py first to download the files.")
        sys.exit(1)

    # Connect to Dropbox
    print("Connecting to Dropbox...")
    dbx, display_name = get_dropbox_client()

    print(f"Authenticated as: {display_name}")
    print(f"Uploading to shared folder: {DROPBOX_BASE_PATH}\n")

    # Ensure destination folders exist
    dropbox_images_path = f"{DROPBOX_BASE_PATH}/images"
    dropbox_pdfs_path = f"{DROPBOX_BASE_PATH}/pdfs"

    print("Ensuring Dropbox folders exist...")
    ensure_folder_exists(dbx, dropbox_images_path)
    ensure_folder_exists(dbx, dropbox_pdfs_path)
    print()

    # Build hash index of existing remote files for content dedup
    remote_hashes = {}  # content_hash -> filename
    print("Building hash index of existing Dropbox files...")
    for folder_path in [dropbox_images_path, dropbox_pdfs_path]:
        try:
            result = dbx.files_list_folder(folder_path)
            entries = result.entries
            while result.has_more:
                result = dbx.files_list_folder_continue(result.cursor)
                entries.extend(result.entries)
            for entry in entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.content_hash:
                    remote_hashes[entry.content_hash] = entry.name
        except ApiError:
            pass
    print(f"Indexed {len(remote_hashes)} existing file(s)\n")

    stats = {"uploaded": 0, "skipped": 0, "duplicate": 0, "failed": 0}

    # Upload images
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    print(f"Uploading {len(image_files)} image(s) to {dropbox_images_path}/...")
    for img in image_files:
        try:
            dest = f"{dropbox_images_path}/{img.name}"
            result = upload_file(dbx, img, dest, remote_hashes)
            stats[result] += 1
        except Exception as e:
            print(f"  [error] {img.name}: {e}")
            stats["failed"] += 1

    print()

    # Upload PDFs
    pdf_files = sorted([
        f for f in pdfs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in PDF_EXTENSIONS
    ])
    print(f"Uploading {len(pdf_files)} PDF(s) to {dropbox_pdfs_path}/...")
    for pdf in pdf_files:
        try:
            dest = f"{dropbox_pdfs_path}/{pdf.name}"
            result = upload_file(dbx, pdf, dest, remote_hashes)
            stats[result] += 1
        except Exception as e:
            print(f"  [error] {pdf.name}: {e}")
            stats["failed"] += 1

    # Summary
    print()
    print("=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"Uploaded:    {stats['uploaded']}")
    print(f"Skipped:     {stats['skipped']}")
    print(f"Duplicates:  {stats['duplicate']}")
    print(f"Failed:      {stats['failed']}")
    print(f"\nDropbox folder: {DROPBOX_SHARED_FOLDER_URL}")


if __name__ == "__main__":
    main()
