import os
from PIL import Image
import torchvision.transforms as transforms

def get_augmentation_pipeline():
    """
    Define a simpler, gentler augmentation pipeline for white-on-black line images.
    """
    augmentation_pipeline = transforms.Compose([
        # Smaller random rotation (±5°) instead of ±15°
        transforms.RandomRotation(degrees=5),
        
        # Random horizontal flip with a lower probability
        transforms.RandomHorizontalFlip(p=0.3),
        
        # Mild color jitter: only brightness & contrast; no saturation/hue
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.0,  # effectively disable saturation changes
            hue=0.0          # disable hue changes
        ),
        
        # Finally, force EXACT 79×68 (height×width)
        transforms.Resize((79, 68))
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
    
    for filename in os.listdir(source_dir):
        # Only process typical image extensions
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(source_dir, filename)
            try:
                # Convert to 'RGB' so transforms.ColorJitter etc. won't complain
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error opening {image_path}: {e}")
                continue

            base_filename, ext = os.path.splitext(filename)
            
            # Generate multiple augmented versions for each image
            for i in range(num_augments):
                augmented_image = augmentation_pipeline(image)
                
                # Create a unique filename for each augmented image
                output_filename = f"{base_filename}_aug{i}{ext}"
                output_path = os.path.join(source_dir, output_filename)
                
                # Save the augmented image
                augmented_image.save(output_path)
                print(f"Saved augmented image: {output_path}")

if __name__ == '__main__':
    source_directory = os.path.dirname(os.path.abspath(__file__))
    num_augmentations = 5
    
    augment_images_inplace(source_directory, num_augmentations)