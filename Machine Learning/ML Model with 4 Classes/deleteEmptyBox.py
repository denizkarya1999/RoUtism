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
    Determines if the image at `image_path` contains a drawn line using Canny edge
    detection and the Probabilistic Hough Line Transform.

    Parameters:
        image_path (str): Path to the image file.
        canny_threshold1 (int): First threshold for Canny edge detection.
        canny_threshold2 (int): Second threshold for Canny edge detection.
        hough_threshold (int): Accumulator threshold for the Hough transform.
        min_line_length (int): Minimum line length for detection.
        max_line_gap (int): Maximum gap allowed between line segments.

    Returns:
        bool: True if at least one line is detected, False if the image is an empty box.
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error reading image: {image_path}")
        return False

    # Detect edges using Canny edge detector
    edges = cv2.Canny(image, canny_threshold1, canny_threshold2)
    
    # Detect lines using the Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    
    return (lines is not None and len(lines) > 0)

def delete_associated_files(line_folder, video_slice_folder):
    """
    Scans the `line_folder` for images whose filenames start with "line_". For each image,
    it checks if the image is an empty box (i.e. no drawn line is detected).
    If an empty box is found, then the function deletes both the empty box image and the
    associated video slice in `video_slice_folder`. The association is based on file name;
    for example, if a file is named "line_1.jpg", the script looks for "videoslice_1.*" in
    the video slice folder.

    Parameters:
        line_folder (str): Directory containing the line (or empty box) images.
        video_slice_folder (str): Directory containing the sliced video images.
    """
    # Supported image file extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

    # Get all images in the line folder that start with "line_"
    all_line_images = []
    for ext in image_extensions:
        pattern = os.path.join(line_folder, ext)
        all_line_images.extend(glob.glob(pattern))
    line_image_files = [f for f in all_line_images if os.path.basename(f).lower().startswith("line_")]

    print(f"Found {len(line_image_files)} line image(s) in {line_folder}.")

    for line_path in line_image_files:
        if not contains_line(line_path):
            # The line image is an empty box so we want to delete both this file
            # and the associated video slice.
            basename = os.path.basename(line_path)
            
            # Assuming the filename is of the form "line_<identifier>.<ext>",
            # we extract the identifier (e.g., "1" in "line_1.jpg").
            try:
                identifier = basename.split('_', 1)[1].split('.')[0]
            except IndexError:
                print(f"Filename format unexpected for file: {basename}. Skipping.")
                continue

            # Build a pattern to search for the associated video slice.
            # It should match something like "videoslice_<identifier>.*"
            video_slice_pattern = os.path.join(video_slice_folder, f"videoslice_{identifier}.*")
            associated_video_files = glob.glob(video_slice_pattern)

            # Delete the associated video slice file(s) if found
            if associated_video_files:
                for video_path in associated_video_files:
                    try:
                        os.remove(video_path)
                        print(f"Deleted associated video slice: {video_path}")
                    except Exception as e:
                        print(f"Error deleting {video_path}: {e}")
            else:
                print(f"No associated video slice found for identifier: {identifier}")

            # Delete the line image (empty box)
            try:
                os.remove(line_path)
                print(f"Deleted empty line image: {line_path}")
            except Exception as e:
                print(f"Error deleting {line_path}: {e}")
        else:
            print(f"Kept line image (line detected): {line_path}")

if __name__ == "__main__":
    import sys

    # Let the user input the folder paths for line images and video slices.
    line_folder = input("Enter the folder path for line images: ").strip()
    video_slice_folder = input("Enter the folder path for video slice images: ").strip()

    # Validate the folders exist.
    if not os.path.isdir(line_folder):
        print(f"Error: '{line_folder}' is not a valid directory.")
        sys.exit(1)
    if not os.path.isdir(video_slice_folder):
        print(f"Error: '{video_slice_folder}' is not a valid directory.")
        sys.exit(1)

    # Execute deletion based on detected empty boxes.
    delete_associated_files(line_folder, video_slice_folder)