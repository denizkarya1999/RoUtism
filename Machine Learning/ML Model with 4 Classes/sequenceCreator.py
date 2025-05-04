#!/usr/bin/env python3
"""
sequenceCreator.py

Usage:
  1. Place this file in a folder that contains images (e.g., .jpg, .png).
  2. Run: python sequenceCreator.py
  3. It will create a "sequences/" folder in the same directory, 
     splitting the images into subfolders of size SEQUENCE_SIZE.

No command-line arguments are needed.
"""

import os
import math
import shutil

# Adjust SEQUENCE_SIZE if you want a different chunk size
SEQUENCE_SIZE = 15

def main():
    # 1) Identify the current folder where the script is placed
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2) The input dir is the same folder as the script
    #    We'll gather images from this folder
    input_dir = script_dir

    # 3) The output dir is "sequences/" within the same folder
    output_dir = os.path.join(script_dir, "sequences")
    os.makedirs(output_dir, exist_ok=True)

    # 4) Gather images
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(input_dir, f))
    ])

    total_images = len(image_files)
    if total_images == 0:
        print("No image files found in this directory. Exiting.")
        return

    print(f"Found {total_images} images in '{input_dir}'")
    print(f"Splitting them into sequences of {SEQUENCE_SIZE} frames each...")

    # 5) Calculate how many sequences we need
    num_sequences = math.ceil(total_images / SEQUENCE_SIZE)

    # 6) Chunk images in increments of SEQUENCE_SIZE
    start_idx = 0
    seq_count = 0

    while start_idx < total_images:
        end_idx = min(start_idx + SEQUENCE_SIZE, total_images)
        seq_images = image_files[start_idx:end_idx]

        seq_folder_name = f"seq{seq_count:03d}"
        seq_folder_path = os.path.join(output_dir, seq_folder_name)
        os.makedirs(seq_folder_path, exist_ok=True)

        # Copy images into the seq folder
        for img_name in seq_images:
            src_path = os.path.join(input_dir, img_name)
            dst_path = os.path.join(seq_folder_path, img_name)
            shutil.copy2(src_path, dst_path)

        print(f"  Created '{seq_folder_name}' with {len(seq_images)} image(s).")

        start_idx = end_idx
        seq_count += 1

    print(f"Done! Created {seq_count} sequence subfolders in '{output_dir}'.")

if __name__ == "__main__":
    main()