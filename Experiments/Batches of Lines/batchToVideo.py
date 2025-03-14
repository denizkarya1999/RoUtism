import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    """Read an image from a file with a Unicode path."""
    try:
        # Read file as a byte array then decode into an image
        file_bytes = np.fromfile(filename, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flags)
        return img
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        return None

def main():
    # Hide the root Tk window
    root = tk.Tk()
    root.withdraw()

    # Ask user to select the folder with photos
    input_dir = filedialog.askdirectory(title="Select Folder Containing Photos")
    if not input_dir:
        print("No input folder selected. Exiting.")
        return

    # Ask user to select the output file (video)
    output_path = filedialog.asksaveasfilename(
        defaultextension=".mp4",
        title="Select Destination for Video",
        filetypes=[("MP4 files", "*.mp4")]
    )
    if not output_path:
        print("No output file selected. Exiting.")
        return

    # Valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    # Get list of image files from the input directory
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_extensions)]
    image_files.sort()  # sort images alphabetically

    # Ensure we have exactly 600 images for 5 minutes at 2 fps
    if len(image_files) < 600:
        print(f"Not enough images found (found {len(image_files)} images). Need at least 600 images.")
        return
    image_files = image_files[:600]

    # Read the first image to determine frame dimensions
    first_frame = imread_unicode(image_files[0])
    if first_frame is None:
        print("Error reading the first image.")
        return
    height, width, _ = first_frame.shape

    # Define the VideoWriter object with fps=2 (0.5 sec per frame)
    fps = 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through each image and write to the video
    for idx, img_path in enumerate(image_files, start=1):
        frame = imread_unicode(img_path)
        if frame is None:
            print(f"Error reading image: {img_path}. Skipping.")
            continue

        # Resize if necessary so that all frames have the same dimensions
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)
        print(f"Processed image {idx}/600", end='\r')

    # Release the video writer
    video_writer.release()
    print(f"\nVideo successfully saved to: {output_path}")

if __name__ == "__main__":
    main()