import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn

# Define a separate class that handles saving the model to a .pth file.
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

def main():
    # -------------------------------------------------------------------------
    # 0. OPTIONAL: ENABLE CUDNN BENCHMARK FOR FASTER TRAINING
    # -------------------------------------------------------------------------
    cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # 1. DATA TRANSFORMS
    # -------------------------------------------------------------------------
    # Random rotation, affine transformations, flips, etc., for augmentation.
    data_transforms = transforms.Compose([
        transforms.RandomRotation(15),             # Rotate up to ±15 degrees
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Shift up to 10% in x and y
            scale=(0.8, 1.2),      # Scale between 80% and 120%
            shear=10               # Shear angle
        ),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # For validation/testing, typically we only do resizing and normalization.
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -------------------------------------------------------------------------
    # 2. DATASET LOADING
    # -------------------------------------------------------------------------
    # If you only have a single folder of 1,568 images, we perform a random split.
    data_dir = 'data'
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    # For a dataset of 1,568 images, we split it into 80% training and 20% validation.
    total_size = len(full_dataset)  # ~1,568 images
    val_size = int(0.2 * total_size)  # ~313 images (approx.)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Override the validation dataset's transform to use the validation transforms
    val_dataset.dataset.transform = val_transforms

    # -------------------------------------------------------------------------
    # 3. CREATE DATALOADERS
    # -------------------------------------------------------------------------
    batch_size = 64
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    class_names = full_dataset.classes
    
    print("Classes detected:", class_names)
    print("Training images:", dataset_sizes['train'])
    print("Validation images:", dataset_sizes['val'])

    # -------------------------------------------------------------------------
    # 4. LOAD & MODIFY RESNET18
    # -------------------------------------------------------------------------
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # -------------------------------------------------------------------------
    # 5. PREPARE FOR TRAINING
    # -------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Reduced number of epochs to 20
    num_epochs = 20

    # -------------------------------------------------------------------------
    # (OPTIONAL) MIXED-PRECISION TRAINING SETUP
    # -------------------------------------------------------------------------
    use_mixed_precision = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    # -------------------------------------------------------------------------
    # 6. TRAINING LOOP
    # -------------------------------------------------------------------------
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    print('Training complete')

    # -------------------------------------------------------------------------
    # 7. SAVE THE TRAINED MODEL
    # -------------------------------------------------------------------------
    exporter = ModelExporter(model, class_names)
    save_path = os.path.join("saved_models", "resnet18_custom.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    exporter.save(save_path)

if __name__ == '__main__':
    main()