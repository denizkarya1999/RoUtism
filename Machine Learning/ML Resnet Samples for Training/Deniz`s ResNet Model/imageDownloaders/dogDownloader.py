import requests
import os
import time

# Create a directory to save dog images
save_folder = 'dog_images'
os.makedirs(save_folder, exist_ok=True)

num_images = 100

for i in range(num_images):
    try:
        # Fetch a random dog image URL from Dog CEO API
        api_url = "https://dog.ceo/api/breeds/image/random"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # The image URL is contained in the "message" field
        image_url = data.get("message")
        if not image_url:
            print(f"No image URL returned for image {i}")
            continue
        
        # Determine file extension from the image URL
        extension = image_url.split('.')[-1].split('?')[0]
        filename = f"dog_{i}.{extension}"
        file_path = os.path.join(save_folder, filename)
        
        # Download the image content
        img_response = requests.get(image_url, timeout=10)
        img_response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(img_response.content)
        
        print(f"Downloaded: {file_path}")
        
        # Pause briefly to avoid overwhelming the API
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error downloading image {i}: {e}")

print("Finished downloading 100 dog images.")
