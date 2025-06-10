import os
import shutil
import random

def organize_train_val_emotions(
    log_file='emotions_log.txt',
    lines_folder='Lines',
    classes_folder='Classes',
    train_ratio=0.7
):
    """
    1) Prompts for user name.
    2) Reads `emotions_log.txt` (skips the header).
    3) For each line (e.g. '0.00,sad'), maps to `Line (1).jpg`, etc.
    4) Renames the image to <UserName>_Line(X).jpg
    5) Collects images by emotion in memory.
    6) Splits them into train and val sets (70/30) *for each emotion*.
    7) Moves them into:
        Classes/train/<emotion>/
        Classes/val/<emotion>/
    """

    # 1) Ask for the user's name
    user_name = input("Enter your name: ").strip()
    if not user_name:
        user_name = "User"  # fallback if user presses enter

    # Ensure the lines folder exists
    if not os.path.exists(lines_folder):
        print(f"Lines folder '{lines_folder}' does not exist. Exiting.")
        return

    # Read the log file
    with open(log_file, 'r') as f:
        lines = f.read().strip().split('\n')

    if len(lines) <= 1:
        print("Log file has no data lines (only header or empty). Exiting.")
        return

    header = lines[0]         # "Timestamp(s),Dominant Emotion"
    data_lines = lines[1:]    # subsequent lines with actual data

    # Dictionary to accumulate files by emotion
    # e.g. emotion_to_files["sad"] = [ (src_path1, renamed_filename1), ... ]
    emotion_to_files = {}

    # 2) Map each data line to the corresponding "Line (index).jpg"
    for i, line in enumerate(data_lines, start=1):
        # Example line: "0.00,sad"
        parts = line.split(',')
        if len(parts) < 2:
            emotion = "misc"
        else:
            emotion = parts[1].strip() or "misc"

        # Source filename: "Line (i).jpg"
        src_filename = f"Line ({i}).jpg"
        src_path = os.path.join(lines_folder, src_filename)

        if not os.path.exists(src_path):
            print(f"File not found: {src_path}. Skipping.")
            continue

        # New name: <UserName>_Line(i).jpg
        new_filename = f"{user_name}_Line({i}).jpg"

        # Accumulate in dictionary
        if emotion not in emotion_to_files:
            emotion_to_files[emotion] = []
        emotion_to_files[emotion].append((src_path, new_filename))

    # 3) For each emotion, split files into train / val
    # Create the directories "Classes/train/<emotion>" and "Classes/val/<emotion>" and move the files
    for emotion, file_info_list in emotion_to_files.items():
        # e.g. file_info_list = [(src_path, new_filename), ...]

        # Shuffle so we get random distribution
        random.shuffle(file_info_list)

        total = len(file_info_list)
        split_idx = int(total * train_ratio)
        train_list = file_info_list[:split_idx]
        val_list   = file_info_list[split_idx:]

        # Create folder structure:
        #   Classes/train/<emotion>/
        #   Classes/val/<emotion>/
        emotion_train_folder = os.path.join(classes_folder, "train", emotion)
        emotion_val_folder   = os.path.join(classes_folder, "val", emotion)
        os.makedirs(emotion_train_folder, exist_ok=True)
        os.makedirs(emotion_val_folder, exist_ok=True)

        # Move train set
        for src_path, renamed_file in train_list:
            dst_path = os.path.join(emotion_train_folder, renamed_file)
            print(f"Moving to train: {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)

        # Move val set
        for src_path, renamed_file in val_list:
            dst_path = os.path.join(emotion_val_folder, renamed_file)
            print(f"Moving to val: {src_path} -> {dst_path}")
            shutil.move(src_path, dst_path)

    print("Dataset organized into train/val folders by emotion.")

if __name__ == "__main__":
    organize_train_val_emotions(
        log_file='emotions_log.txt',  # If your log is named differently, change here
        lines_folder='Lines',         # Folder with 'Line (1).jpg', 'Line (2).jpg', ...
        classes_folder='Classes',      # We'll create Classes/train/emotion and Classes/val/emotion
        train_ratio=0.7               # 70% train, 30% val
    )