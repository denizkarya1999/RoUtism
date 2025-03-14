import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def load_and_prepare_image(image_path, target_channels=3):
    """Load image and ensure it has the required number of channels (BGR)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    # If image is grayscale, convert to BGR
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def resize_to_common_height(img1, img2):
    """Resize both images to have the same height (minimum of the two) while maintaining aspect ratios."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    common_height = min(h1, h2)
    
    # Compute new widths maintaining aspect ratio
    new_w1 = int(w1 * common_height / h1)
    new_w2 = int(w2 * common_height / h2)
    
    resized_img1 = cv2.resize(img1, (new_w1, common_height))
    resized_img2 = cv2.resize(img2, (new_w2, common_height))
    
    return resized_img1, resized_img2

def combine_images(video_img, line_img):
    """Combine two images side by side."""
    # Resize both images to the same height
    video_resized, line_resized = resize_to_common_height(video_img, line_img)
    # Concatenate images horizontally
    combined = np.hstack((video_resized, line_resized))
    return combined

def main():
    # Set up Tkinter root and hide the main window
    root = tk.Tk()
    root.withdraw()

    # Prompt user to select exactly 600 video images
    messagebox.showinfo("Select Files", "Please select EXACTLY 600 video images.")
    video_paths = filedialog.askopenfilenames(
        title="Select 600 Video Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    if not video_paths:
        print("No video images selected. Exiting.")
        return
    if len(video_paths) != 600:
        messagebox.showwarning("Selection Error", f"Expected 600 video images, but selected {len(video_paths)}. Exiting.")
        return

    # Prompt user to select exactly 600 line images
    messagebox.showinfo("Select Files", "Please select EXACTLY 600 line images.")
    line_paths = filedialog.askopenfilenames(
        title="Select 600 Line Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    if not line_paths:
        print("No line images selected. Exiting.")
        return
    if len(line_paths) != 600:
        messagebox.showwarning("Selection Error", f"Expected 600 line images, but selected {len(line_paths)}. Exiting.")
        return

    # Optionally sort the file lists to ensure consistent pairing
    video_paths = sorted(video_paths)
    line_paths = sorted(line_paths)

    # Ask for an output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        print("No output folder selected. Exiting.")
        return

    # Process each pair of images and create the combined image
    for i, (video_path, line_path) in enumerate(zip(video_paths, line_paths)):
        try:
            video_img = load_and_prepare_image(video_path)
            line_img = load_and_prepare_image(line_path)
        except ValueError as e:
            print(e)
            continue

        combined_img = combine_images(video_img, line_img)

        # Save the combined image
        output_filename = os.path.join(output_folder, f"association_{i:04d}.jpg")
        cv2.imwrite(output_filename, combined_img)
        print(f"Saved {output_filename}")

    messagebox.showinfo("Done", "Finished creating associated images.")

if __name__ == "__main__":
    main()
