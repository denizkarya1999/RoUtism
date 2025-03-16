import os
from rembg import remove
from tkinter import Tk, filedialog
from PIL import Image
import io

def select_input_files():
    # Open file dialog to select multiple image files
    Tk().withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    return file_paths

def select_output_directory():
    # Open file dialog to select output directory
    Tk().withdraw()
    output_directory = filedialog.askdirectory(title="Select Output Directory")
    return output_directory

def process_images(input_files, output_dir):
    for file_path in input_files:
        try:
            # Read the input image
            with open(file_path, "rb") as input_file:
                input_image = input_file.read()
                
            # Remove background
            output_image_data = remove(input_image)
            
            # Convert the output image to a format that PIL can handle
            output_image = Image.open(io.BytesIO(output_image_data))
            
            # Convert RGBA to RGB (if necessary)
            if output_image.mode == 'RGBA':
                output_image = output_image.convert('RGB')
            
            # Save the output image
            base_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, base_name)
            
            # If the file was originally a JPEG, save it as JPEG; otherwise, save as PNG
            if base_name.lower().endswith('.jpg') or base_name.lower().endswith('.jpeg'):
                output_image.save(output_path, 'JPEG')
            else:
                output_image.save(output_path, 'PNG')
            
            print(f"Processed {base_name} and saved to {output_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    input_files = select_input_files()
    if not input_files:
        print("No input files selected. Exiting...")
        return
    
    output_dir = select_output_directory()
    if not output_dir:
        print("No output directory selected. Exiting...")
        return
    
    process_images(input_files, output_dir)

if __name__ == "__main__":
    main()