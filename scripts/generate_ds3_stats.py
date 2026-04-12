import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def count_files(dir_path):
    if not dir_path.exists():
        return 0
    return len([f for f in dir_path.iterdir() if f.is_file() and f.suffix in ['.pdf', '.txt']])

def main():
    base_dir = Path("/Users/guest2/Desktop/repos/talentlens/TalentLens_Public")
    
    # 1. Count new extraction
    new_member_dir = base_dir / "data" / "ds3" / "member_resumes"
    new_board_dir = base_dir / "data" / "ds3" / "board_resumes"
    
    new_members = count_files(new_member_dir)
    new_board = count_files(new_board_dir)
    
    # 2. Count failures
    failure_log_path = base_dir / "failed_extractions_report.csv"
    failures = 0
    if failure_log_path.exists():
        with open(failure_log_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # subtract header
            try:
                next(reader)
                failures = sum(1 for row in reader)
            except StopIteration:
                pass
                
    # 3. Count existing in test/ (if any)
    existing_board_dir = base_dir / "test" / "board"
    existing_member_dir = base_dir / "test" / "members"
    
    existing_board = count_files(existing_board_dir)
    existing_members = count_files(existing_member_dir)

    # Since the new CSV didn't have board vs member distinction, we saved all 
    # to data/ds3/member_resumes. But the user wants the "number of board number of members".
    # Total members = new_members + existing_members
    # Total board = new_board + existing_board
    
    total_board = new_board + existing_board
    total_members = new_members + existing_members
    total_ds3 = total_board + total_members
    
    print(f"Total DS3 Resumes (Board + Members): {total_ds3}")
    print(f"Total Board: {total_board}")
    print(f"Total Members: {total_members}")
    print(f"Total Failed Extractions: {failures}")
    
    # Generate bar chart
    labels = ['Board', 'Members', 'Failed']
    values = [total_board, total_members, failures]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#4C72B0', '#55A868', '#C44E52'])
    plt.title('Talent Pool Resume Pipeline Statistics')
    plt.ylabel('Count')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')
                 
    output_img = Path("/Users/guest2/.gemini/antigravity/artifacts/resume_stats.png")
    output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img)
    plt.close()
    print(f"Saved bar chart to {output_img}")

if __name__ == "__main__":
    main()
