# TalentLens_Public
NLP pipeline for public resume dataset for TalentLens

## Discord Resume Data

Resume images and PDFs collected from Discord are stored on a shared Dropbox folder (not in the git repo due to file size).

**Dropbox folder:** [Talentlens/Discord_Resumes](https://www.dropbox.com/scl/fo/jepsuisabfchr3mn70v3v/AJ6PRx4-b0IURZ_Lb_Ox56E?rlkey=uqsahacirzq5noxhlrmbeefdp&st=za2smdwe&dl=0)

---

### Downloading Files from Dropbox

Copy the Dropbox shared folder link and run:

```bash
cd data/discord
python download_from_dropbox.py "<dropbox-folder-link>"
```

Example:
```bash
python download_from_dropbox.py "https://www.dropbox.com/scl/fo/jepsuisabfchr3mn70v3v/AJ6PRx4-b0IURZ_Lb_Ox56E?rlkey=uqsahacirzq5noxhlrmbeefdp&st=za2smdwe&dl=0"
```

This will automatically download and extract files into `data/discord/images/` and `data/discord/pdfs/`. No API token or Dropbox account needed.

---

### Uploading Files to Dropbox

Uploading requires a Dropbox API token (one-time setup).

#### Setup

1. Install dependencies:
   ```bash
   pip install dropbox python-dotenv
   ```

2. Create a Dropbox app at https://www.dropbox.com/developers/apps:
   - **Create app** → **Scoped access** → **Full Dropbox**
   - **Permissions** tab → enable `files.content.write`, `files.content.read`, `sharing.write` → **Submit**
   - **Settings** tab → click **Generate** under "Generated access token"

3. Paste the token in `data/discord/.env`:
   ```
   DROPBOX_ACCESS_TOKEN=paste_your_token_here
   ```
   > Tokens expire after ~4 hours. Generate a new one when expired.

#### Run

```bash
cd data/discord
python upload_to_dropbox.py
```

This uploads `images/` and `pdfs/` to the shared `Talentlens/Discord_Resumes/` folder, skipping files that already exist.

---

### File Structure

```
data/discord/
├── links/                      # .webloc files with Discord CDN URLs
├── images/                     # Downloaded resume images (git-ignored)
├── pdfs/                       # Downloaded resume PDFs (git-ignored)
├── .env                        # Dropbox access token (git-ignored)
├── download_discord_images.py  # Downloads files from Discord CDN links
├── download_from_dropbox.py    # Downloads files from any Dropbox shared link
├── upload_to_dropbox.py        # Uploads files to shared Dropbox folder
└── data.ipynb                  # Data exploration notebook
```
