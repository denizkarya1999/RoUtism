import os
import cv2
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.simpledialog import askstring

# Hide the root Tk window
root = Tk()
root.withdraw()

# Let the user select the folder containing raw images
input_folder = askdirectory(title="Select Folder where raw images are stored")
if not input_folder:
    raise ValueError("No input folder was selected.")

# Ask the user for the emotion type (e.g., happy, sad, etc.)
emotion = askstring("Emotion", "Enter the emotion type (e.g., happy, sad, etc.):")
if not emotion:
    raise ValueError("No emotion type provided!")

# Let the user select the folder where the converted folder will be saved
save_base_folder = askdirectory(title="Select Folder to Save the Converted Data Folder")
if not save_base_folder:
    raise ValueError("No folder selected for saving the converted data.")

# Create the output folder named "<emotion>_converted" in the selected save folder
output_folder = os.path.join(save_base_folder, f"{emotion}_converted")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image: convert to grayscale and resize to 28x28
files = sorted(os.listdir(input_folder))
for index, file_name in enumerate(files, start=1):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(input_folder, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image {file_name}. Skipping.")
            continue

        # Resize image to 28x28
        img_resized = cv2.resize(img, (28, 28))

        # Create new file name using the emotion label and index
        file_ext = os.path.splitext(file_name)[1]
        new_file_name = f"{emotion}_{index}{file_ext}"
        output_path = os.path.join(output_folder, new_file_name)

        # Save the processed image in the output folder
        cv2.imwrite(output_path, img_resized)
        print(f"Processed and saved: {new_file_name}")

print("All images have been converted to 28x28 and saved in:", output_folder)