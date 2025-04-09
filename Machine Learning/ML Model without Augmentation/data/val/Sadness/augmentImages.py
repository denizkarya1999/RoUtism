import tkinter as tk
from tkinter import filedialog
import Augmentor

def main():
    # Hide the main tkinter root window
    root = tk.Tk()
    root.withdraw()
    
    # Prompt the user to select the input directory
    input_dir = filedialog.askdirectory(title="Select the input directory containing images")
    
    # If the user cancels the dialog, input_dir will be empty
    if not input_dir:
        print("No directory selected. Exiting.")
        return
    
    # Create an Augmentor pipeline
    p = Augmentor.Pipeline(input_dir)
    
    # Add augmentation operations
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.zoom(probability=0.3, min_factor=1.1, max_factor=1.5)
    
    # Generate 2400 augmented images
    p.sample(100)
    
    print(f"Augmented images saved to: {input_dir}/output")

if __name__ == "__main__":
    main()