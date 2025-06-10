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
    # Define data transforms for training and validation.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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
    
    # Assume your data directory is structured as follows:
    # data/
    #   train/
    #     deniz/    (e.g., 80 images)
    #     cat/      (e.g., 80 images)
    #     dogs/     (e.g., 80 images)
    #     birds/     (e.g., 80 images)
    #   val/
    #     deniz/    (e.g., 20 images)
    #     cat/      (e.g., 20 images)
    #     dogs/     (e.g., 20 images)
    #     birds/     (e.g., 20 images)
    data_dir = 'data'  # Change this to your dataset root directory
    
    # Create datasets for training and validation.
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                transform=data_transforms[x])
                      for x in ['train', 'val']}
    
    # Create dataloaders with num_workers=0 for Windows compatibility.
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    
    # Get dataset sizes and class names.
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print("Classes detected:", class_names)
    print("Training images:", dataset_sizes['train'])
    print("Validation images:", dataset_sizes['val'])
    
    # Load the pre-trained ResNet18 model.
    model = models.resnet18(pretrained=True)
    
    # Modify the final fully connected layer to match the number of classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Move the model to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Set the number of epochs.
    num_epochs = 25
    
    # Training loop.
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
    
        # Each epoch has a training and validation phase.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode.
            else:
                model.eval()   # Set model to evaluation mode.
    
            running_loss = 0.0
            running_corrects = 0
    
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                # Zero the parameter gradients.
                optimizer.zero_grad()
    
                # Forward pass (track gradients only in training phase).
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
    
    # Instantiate the ModelExporter and save the model.
    exporter = ModelExporter(model, class_names)
    save_path = os.path.join("saved_models", "resnet18_custom.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    exporter.save(save_path)

if __name__ == '__main__':
    main()
