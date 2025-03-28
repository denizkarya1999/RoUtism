import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
from PIL import Image

# -----------------------------------------------------------------------------
# 1. MODEL EXPORTER: Saves model state and class names to a .pth file.
# -----------------------------------------------------------------------------
class ModelExporter:
    def __init__(self, model, class_names):
        """
        Args:
            model: Trained PyTorch model.
            class_names: List of class names.
        """
        self.model = model
        self.class_names = class_names

    def save(self, file_path):
        """Save the model's state dictionary along with class names."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names
        }
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")

# -----------------------------------------------------------------------------
# 2. CUSTOM DATASET: Define your own dataset to load images from custom classes.
# -----------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        """
        Args:
            data_dir (str): Directory with all the images organized in subfolders.
            class_names (list): List of class names.
            transform: Transformations to apply to the images.
        """
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Expecting a directory structure: data_dir/class_name/image.jpg
        for idx, class_name in enumerate(class_names):
            class_folder = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_folder):
                print(f"Warning: {class_folder} does not exist.")
                continue
            for img_file in os.listdir(class_folder):
                if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(class_folder, img_file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------------------------------------------------------
# 3. MAIN FUNCTION: Data transforms, dataset loading, model training, and saving.
# -----------------------------------------------------------------------------
def main():
    # -------------------------------------------------------------------------
    # 0. ENABLE CUDNN BENCHMARK FOR FASTER TRAINING
    # -------------------------------------------------------------------------
    cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # 1. DATA TRANSFORMS
    # -------------------------------------------------------------------------
    # Enhanced training transforms: added RandomHorizontalFlip for better augmentation.
    data_transforms = transforms.Compose([
        transforms.RandomRotation(15),             # Rotate up to ±15 degrees
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Shift up to 10% in x and y
            scale=(0.8, 1.2),      # Scale between 80% and 120%
            shear=10               # Shear angle
        ),
        transforms.RandomHorizontalFlip(),           # Flip images horizontally at random
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -------------------------------------------------------------------------
    # 2. DATASET LOADING (USING THE ENTIRE DATASET FOR TRAINING)
    # -------------------------------------------------------------------------
    custom_classes = ['Angry', 'Anxiety', 'Excitement', 'Sadness']
    data_dir = 'data'  # This folder should contain subfolders for each class.
    dataset = CustomDataset(data_dir, custom_classes, transform=data_transforms)

    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("Classes detected:", custom_classes)
    print("Total images:", len(dataset))

    # -------------------------------------------------------------------------
    # 3. LOAD & MODIFY RESNET18
    # -------------------------------------------------------------------------
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(custom_classes))

    # -------------------------------------------------------------------------
    # 4. PREPARE FOR TRAINING
    # -------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Use Adam optimizer with a lower learning rate and weight decay for regularization.
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Cosine annealing scheduler for smoother learning rate decay.
    num_epochs = 20  # Increased number of epochs for more training iterations.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Optional: Mixed-precision training setup.
    use_mixed_precision = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    # -------------------------------------------------------------------------
    # 5. TRAINING LOOP (ONLY TRAINING PHASE)
    # -------------------------------------------------------------------------
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    print('Training complete')

    # -------------------------------------------------------------------------
    # 6. SAVE THE TRAINED MODEL
    # -------------------------------------------------------------------------
    exporter = ModelExporter(model, custom_classes)
    save_path = os.path.join("saved_models", "resnet18_custom.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    exporter.save(save_path)

if __name__ == '__main__':
    main()
