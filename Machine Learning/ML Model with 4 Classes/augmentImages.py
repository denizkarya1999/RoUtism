import os
from PIL import Image
import torchvision.transforms as transforms

def get_augmentation_pipeline():
    """Define the augmentation pipeline."""
    augmentation_pipeline = transforms.Compose([
        transforms.RandomRotation(15),             # Random rotation within ±15 degrees
        transforms.RandomHorizontalFlip(),           # Random horizontal flip
        transforms.ColorJitter(                      # Random adjustments for brightness, contrast, saturation, and hue
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomResizedCrop(224,           # Random crop and resize to 224x224 pixels
                                     scale=(0.8, 1.0),
                                     ratio=(0.9, 1.1))
    ])
    return augmentation_pipeline

def augment_images_inplace(source_dir, num_augments=5):
    """
    Augment images from the source directory and save them in the same folder.
    
    Args:
        source_dir (str): Path to the folder containing the original images.
        num_augments (int): Number of augmented versions to create per image.
    """
    augmentation_pipeline = get_augmentation_pipeline()
    
    # Process each image in the source directory.
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(source_dir, filename)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error opening {image_path}: {e}")
                continue

            base_filename, ext = os.path.splitext(filename)
            
            # Generate multiple augmented versions for each image.
            for i in range(num_augments):
                augmented_image = augmentation_pipeline(image)
                # Create a new filename to avoid overwriting the original image.
                output_filename = f"{base_filename}_aug{i}{ext}"
                output_path = os.path.join(source_dir, output_filename)
                augmented_image.save(output_path)
                print(f"Saved augmented image: {output_path}")

if __name__ == '__main__':
    # Use the directory where this source code is located.
    source_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Number of augmentations to create per image.
    num_augmentations = 5
    
    augment_images_inplace(source_directory, num_augmentations)
