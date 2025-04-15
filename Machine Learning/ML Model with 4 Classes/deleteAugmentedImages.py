import os

# Get the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate over each file in the current directory
for filename in os.listdir(current_dir):
    # Check if the file name contains the substring "aug"
    if "aug" in filename:
        file_path = os.path.join(current_dir, filename)
        # Only remove if it's actually a file (not a directory)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
