import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def slice_video_to_images(video_path, output_folder, num_images=600):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_images:
        print("Warning: The video has fewer frames than the number of images requested.")
        num_images = total_frames

    # Compute the frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, num=num_images, dtype=int)
    
    # Loop through and save each selected frame
    for count, frame_idx in enumerate(frame_indices):
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Define the filename with zero padded count
            filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        else:
            print(f"Warning: Could not read frame at index {frame_idx}")

    cap.release()
    print("Finished slicing video into images.")

def main():
    # Set up Tkinter root and hide the main window
    root = tk.Tk()
    root.withdraw()

    # Prompt the user to select a video file
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
    )
    if not video_path:
        print("No video file selected. Exiting.")
        return

    # Prompt the user to select an output folder for the images
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        print("No output folder selected. Exiting.")
        return

    # Process the video to extract and save images
    slice_video_to_images(video_path, output_folder)

if __name__ == "__main__":
    main()
