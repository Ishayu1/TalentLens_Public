from pathlib import Path

logfile = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public/manually_need_to_add.txt")

if not logfile.exists():
    print("Log file not found.")
    exit()

with open(logfile, 'r') as f:
    content = f.read()

# Entries are separated by 30 dashes and a newline
entries = content.split("-" * 30 + "\n")

cleaned_entries = []
for entry in entries:
    if not entry.strip():
        continue
    # Check if "Reason: no link given" is in this entry
    if "Reason: no link given" in entry:
        continue
    cleaned_entries.append(entry.strip() + "\n" + "-" * 30 + "\n")

with open(logfile, 'w') as f:
    f.writelines(cleaned_entries)

print(f"Cleaned {len(entries) - len(cleaned_entries)} 'no link given' entries from {logfile.name}")
