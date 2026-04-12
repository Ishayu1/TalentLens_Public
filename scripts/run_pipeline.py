import os
import csv
import subprocess
from pathlib import Path

def setup_members_csv():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    input_csv = base_dir / "Supabase Snippet Talent Pool Resume Links.csv"
    
    out_dir = base_dir / "data" / "ds3" / "member_resumes"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "members.csv"
    
    print("Formatting members CSV to match pipeline requirements...")
    
    header_mapping = {
        'full_name': 'Full Name',
        'major': 'Major',
        'graduation_year': 'Graduation Year',
        'resume_link': 'Resume Link',
        'linkedin_link': 'Linkedin Link',
        'github_link': 'Github Link'
    }
    
    if input_csv.exists():
        with open(input_csv, 'r', encoding='utf-8') as f_in, \
             open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
             
            reader = csv.DictReader(f_in)
            
            # Use original headers mapped to new ones, fallback to original if not found
            new_fieldnames = [header_mapping.get(h, h) for h in reader.fieldnames]
            writer = csv.DictWriter(f_out, fieldnames=new_fieldnames)
            writer.writeheader()
            
            for row in reader:
                new_row = {}
                for old_key, val in row.items():
                    new_key = header_mapping.get(old_key, old_key)
                    new_row[new_key] = val
                writer.writerow(new_row)
                
        print(f"Created configured metadata CSV at {output_csv}")
    else:
        print("[ERROR] Could not find the original CSV!")

def run_script(script_path):
    print(f"\n--- Running {script_path} ---")
    base_dir = "/Users/guest2/Desktop/repos/talentlens/TalentLens_Public"
    python_exec = f"{base_dir}/venv/bin/python"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = base_dir
    
    process = subprocess.Popen(
        [python_exec, script_path],
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        
    process.wait()
    if process.returncode != 0:
        print(f"[ERROR] Script {script_path} failed with return code {process.returncode}")
    else:
        print(f"--- Completed {script_path} ---")

def main():
    setup_members_csv()
    
    base_dir = "/Users/guest2/Desktop/repos/talentlens/TalentLens_Public"
    run_script(f"{base_dir}/src/pipeline/01_text_extraction.py")
    run_script(f"{base_dir}/src/pipeline/ds3_rebuild.py")
    
if __name__ == "__main__":
    main()
