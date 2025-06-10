import os
import shutil

def move_back_to_lines(classes_folder='Classes', lines_folder='Lines'):
    """
    Traverses all subfolders under `classes_folder` (e.g. train/emotion, val/emotion)
    and moves each image back into `lines_folder`.
    
    Renames them as Line (1).jpg, Line (2).jpg, etc.
    
    :param classes_folder: Root folder containing subfolders like train/val/emotion
    :param lines_folder: Destination folder (e.g., "Lines") to place the renamed images
    """
    # Ensure lines_folder exists (create if not)
    if not os.path.exists(lines_folder):
        os.makedirs(lines_folder)
        print(f"Created folder: {lines_folder}")

    # We'll maintain a simple counter for line numbering
    line_counter = 1

    # Walk through all files in Classes folder (recursive)
    for root, dirs, files in os.walk(classes_folder):
        for filename in files:
            # Check if it's an image you want to move (e.g., .jpg, .png, etc.)
            # For simplicity, let's assume .jpg only. Adjust if needed.
            if filename.lower().endswith('.jpg'):
                old_path = os.path.join(root, filename)

                # Construct the new filename as "Line (X).jpg"
                new_filename = f"Line ({line_counter}).jpg"
                line_counter += 1

                new_path = os.path.join(lines_folder, new_filename)

                print(f"Moving '{old_path}' -> '{new_path}'")
                shutil.move(old_path, new_path)

    print("All images have been moved back to the 'Lines' folder.")


if __name__ == "__main__":
    # Adjust folders if needed
    move_back_to_lines(classes_folder='Classes', lines_folder='Lines')
