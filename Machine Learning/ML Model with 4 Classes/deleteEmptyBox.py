import os
import glob
import cv2
import numpy as np

def contains_line(image_path,
                  canny_threshold1=30,
                  canny_threshold2=100,
                  hough_threshold=20,
                  min_line_length=20,
                  max_line_gap=5):
    """
    Determines if the image at `image_path` contains a drawn line by applying
    Canny edge detection and the Probabilistic Hough Line Transform.

    Parameters:
        image_path (str): Path to the image file.
        canny_threshold1 (int): First threshold for Canny edge detection.
        canny_threshold2 (int): Second threshold for Canny edge detection.
        hough_threshold (int): Accumulator threshold for the Hough transform.
        min_line_length (int): Minimum length of line to detect.
        max_line_gap (int): Maximum allowed gap between line segments.

    Returns:
        bool: True if at least one line is detected, False otherwise.
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error reading image: {image_path}")
        return False

    # Detect edges using Canny
    edges = cv2.Canny(image, canny_threshold1, canny_threshold2)
    
    # Use the Probabilistic Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    
    return (lines is not None and len(lines) > 0)

def delete_images(folder_path):
    """
    Scans a folder for image files and deletes any image that either:
      - Does not contain a detected line, or
      - Has a filename containing the substring "aug" (case-insensitive).
    
    Parameters:
        folder_path (str): Path to the folder containing images.
    """
    # Supported image file extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    print(f"Found {len(image_files)} images in {folder_path}.")

    for image_path in image_files:
        filename = os.path.basename(image_path).lower()
        # Delete if filename contains "aug" or if no line is detected
        if "aug" in filename:
            try:
                os.remove(image_path)
                print(f"Deleted image (filename contains 'aug'): {image_path}")
            except Exception as e:
                print(f"Error deleting {image_path}: {e}")
        elif not contains_line(image_path):
            try:
                os.remove(image_path)
                print(f"Deleted image (no line detected): {image_path}")
            except Exception as e:
                print(f"Error deleting {image_path}: {e}")
        else:
            print(f"Kept image: {image_path}")

if __name__ == "__main__":
    # Use the folder where this script is located
    folder_path = os.path.dirname(os.path.abspath(__file__))
    delete_images(folder_path)
