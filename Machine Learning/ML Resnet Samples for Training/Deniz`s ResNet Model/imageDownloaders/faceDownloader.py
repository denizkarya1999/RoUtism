import requests
import os
import time
import tkinter as tk
from tkinter import filedialog
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def select_folder():
    # Open a dialog for the user to select a folder
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder to Save Face Images")
    return folder_selected

def get_session():
    session = requests.Session()
    # Set a custom user agent to mimic a real browser
    session.headers.update({
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/90.0.4430.212 Safari/537.36')
    })
    # Configure retries: 5 retries with exponential backoff for common transient errors
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def download_face_images(save_folder, num_images=100):
    if not save_folder:
        print("No folder selected. Exiting.")
        return

    os.makedirs(save_folder, exist_ok=True)
    # API URL for random user data
    api_url = 'https://randomuser.me/api/'
    session = get_session()

    for i in range(num_images):
        try:
            # Get random user data with a timeout
            response = session.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Extract the URL for the large image
            picture_url = data['results'][0]['picture']['large']
            
            # Determine the file extension (typically jpg)
            extension = picture_url.split('.')[-1].split('?')[0]
            filename = f"face_{i}.{extension}"
            file_path = os.path.join(save_folder, filename)
            
            # Download the face image with a timeout
            img_response = session.get(picture_url, timeout=10)
            img_response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(img_response.content)
            
            print(f"Downloaded: {file_path}")
            # Delay to reduce server load
            time.sleep(1)
        except Exception as e:
            print(f"Error downloading image {i}: {e}")

if __name__ == "__main__":
    folder = select_folder()
    download_face_images(folder)
