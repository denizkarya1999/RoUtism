import os
import shutil
import re

def main():
    # File that lists lines and their associated emotions
    heart_rate_file = 'HeartRate_categorized.txt'
    
    # Folder where "Line (1).jpg", "Line (2).jpg", etc. are stored
    lines_folder = 'Lines'
    
    # Top-level folder for categorized results
    base_categorized_folder = 'Categorized'
    os.makedirs(base_categorized_folder, exist_ok=True)
    
    with open(heart_rate_file, 'r', encoding='utf-8') as f:
        for text_line in f:
            text_line = text_line.strip()
            if not text_line:
                continue  # Skip empty lines

            # Example text line: "Line (1) 22.54.34.000 — 121 bpm — Excited"
            parts = text_line.split('—')
            if len(parts) < 3:
                print(f"Skipping malformed line: {text_line}")
                continue
            
            left_part = parts[0].strip()  # e.g., "Line (1) 22.54.34.000"
            emotion = parts[-1].strip()   # e.g., "Excited"
            
            # Use regex to extract "Line (1)" from left_part
            match_line_name = re.search(r'(Line\s*\(\d+\))', left_part)
            if not match_line_name:
                print(f"Could not find line name in: {left_part}")
                continue
            
            line_name = match_line_name.group(1)  # e.g., "Line (1)"
            
            # Create the emotion subfolder inside Categorized
            emotion_folder = os.path.join(base_categorized_folder, emotion)
            os.makedirs(emotion_folder, exist_ok=True)
            
            # Now, assume each file is "Line (X).jpg"
            source_file_name = line_name + '.jpg'
            source_path = os.path.join(lines_folder, source_file_name)
            
            # Construct the destination path
            destination_path = os.path.join(emotion_folder, source_file_name)
            
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                print(f"Moved '{source_path}' -> '{destination_path}'")
            else:
                print(f"Source file not found: {source_path}")

if __name__ == '__main__':
    main()