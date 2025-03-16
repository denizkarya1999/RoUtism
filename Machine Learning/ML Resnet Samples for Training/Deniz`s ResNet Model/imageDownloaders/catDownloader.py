import requests
import os
import time
import tkinter as tk
from tkinter import filedialog

def select_folder():
    # Open a dialog for the user to select a folder
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title="Select Folder to Save Cat Images")
    return folder_selected

def download_cat_images(save_folder, num_images=100):
    if not save_folder:
        print("No folder selected. Exiting.")
        return

    os.makedirs(save_folder, exist_ok=True)
    api_url = 'https://api.thecatapi.com/v1/images/search'
    
    for i in range(num_images):
        try:
            # Get a random cat image URL
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            image_url = data[0]['url']
            
            # Determine the file extension from the URL
            extension = image_url.split('.')[-1].split('?')[0]
            filename = f"cat_{i}.{extension}"
            file_path = os.path.join(save_folder, filename)
            
            # Download the image content
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(img_response.content)
            
            print(f"Downloaded: {file_path}")
            
            # Brief pause to avoid overwhelming the server
            time.sleep(0.5)
        except Exception as e:
            print(f"Error downloading image {i}: {e}")

if __name__ == "__main__":
    folder = select_folder()
    download_cat_images(folder)