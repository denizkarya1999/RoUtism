import os

def delete_images():
    # Define common image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.svg')
    
    # List all files in the current directory that match the defined extensions
    images = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
    
    # If no images are found, exit the function
    if not images:
        print("No image files found in the current directory.")
        return
    
    # Display the images
    print("Image files in the current directory:")
    for index, img_file in enumerate(images, start=1):
        print(f"{index}. {img_file}")
    
    # Prompt the user for the number of files to delete
    while True:
        try:
            files_to_delete = int(input("\nHow many files do you want to delete? "))
            if files_to_delete < 0:
                print("Please enter a non-negative integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    # Check if the requested number exceeds the number of available images
    if files_to_delete > len(images):
        print(f"\nYou requested to delete {files_to_delete} files, but only {len(images)} are available.")
        print("All image files will be deleted instead.")
        files_to_delete = len(images)

    # Delete the specified number of image files
    for i in range(files_to_delete):
        file_to_remove = images[i]
        os.remove(file_to_remove)
        print(f"Deleted: {file_to_remove}")
    
    print("\nDeletion complete!")

if __name__ == "__main__":
    delete_images()