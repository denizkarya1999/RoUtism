import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

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
    # 1. DATA TRANSFORMS
    # -------------------------------------------------------------------------
    # We apply random rotation, affine transformations, and flips to augment
    # simple line images, giving the model more variety to learn from.
    data_transforms = {
        'train': transforms.Compose([
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
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }
    
    # -------------------------------------------------------------------------
    # 2. DATASET STRUCTURE
    # -------------------------------------------------------------------------
    # Assume you have collected 600 images in total, organized into 9 classes:
    #   Angry, Anxiety, Boredom, Curiosity, Excitement, Jumpscare,
    #   Narcissm, Sadness, Shame
    # in the following structure:
    #
    # data/
    #   train/
    #     Angry/
    #     Anxiety/
    #     Boredom/
    #     Curiosity/
    #     Excitement/
    #     Jumpscare/
    #     Narcissm/
    #     Sadness/
    #     Shame/
    #   val/
    #     Angry/
    #     Anxiety/
    #     Boredom/
    #     Curiosity/
    #     Excitement/
    #     Jumpscare/
    #     Narcissm/
    #     Sadness/
    #     Shame/
    #
    # Where ~80% (480 images) are in 'train/' and ~20% (120 images) are in 'val/'.
    data_dir = 'data'  # Change if your dataset is stored elsewhere.
    
    # -------------------------------------------------------------------------
    # 3. CREATE DATASETS & DATALOADERS
    # -------------------------------------------------------------------------
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print("Classes detected:", class_names)
    print("Training images:", dataset_sizes['train'])
    print("Validation images:", dataset_sizes['val'])
    
    # -------------------------------------------------------------------------
    # 4. LOAD & MODIFY RESNET18
    # -------------------------------------------------------------------------
    # We use a pre-trained ResNet18 on ImageNet and replace its final layer
    # to match the number of classes (9).
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
    
    num_epochs = 25
    
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
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
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