import csv
import os
import re
import shutil

file_path = 'VNKurisuDialogues.csv'
backup_path = 'VNKurisuDialogues_backup.csv'

def clean_text(text):
    # Remove circled numbers (e.g., ⑰, ⑩, etc.)
    text = re.sub(r'[\u2460-\u24ff]', '', text)
    # Strip whitespace
    text = text.strip()
    return text

def clean_data():
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Create backup
    shutil.copy2(file_path, backup_path)
    print(f"Backup created at {backup_path}")

    seen_responses = set()
    cleaned_rows = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("File is empty.")
            return
        
        for row in reader:
            if len(row) < 2:
                continue
            
            name = row[0].strip()
            response = clean_text(row[1])
            
            if not response:
                continue
                
            # Remove repeated lines
            if response in seen_responses:
                continue
            
            seen_responses.add(response)
            cleaned_rows.append([name, response])

    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned_rows)

    print(f"Original file {file_path} has been cleaned.")
    print(f"Total unique lines saved: {len(cleaned_rows)}")

if __name__ == "__main__":
    clean_data()
