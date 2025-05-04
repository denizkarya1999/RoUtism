import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

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
    # 1. Define data transforms for training and validation.
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

    # 2. Specify the root data directory
    data_dir = 'data'  # Adjust path as needed

    # 3. Create ImageFolder datasets for training & validation
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in ['train', 'val']
    }

    # 4. Create DataLoaders
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        for x in ['train', 'val']
    }

    # 5. Gather dataset sizes & class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print("Classes detected:", class_names)
    print("Training images:", dataset_sizes['train'])
    print("Validation images:", dataset_sizes['val'])

    # 6. Load a ResNet model (ResNet-50 here)
    model = models.resnet50(pretrained=True)

    # Replace the final fully connected layer to match the number of classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # 7. Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 8. Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 9. Training loop
    num_epochs = 10

    # Open a file in append mode so we can write logs each epoch
    log_file_path = "training_log.txt"
    with open(log_file_path, "w") as f:
        f.write("Epoch,Phase,Loss,Accuracy\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to eval mode

            running_loss = 0.0
            running_corrects = 0

            # Per-class counters
            class_correct = [0] * len(class_names)
            class_total   = [0] * len(class_names)

            # -------------- Forward Pass --------------
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Only track gradients if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate overall loss & correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # ----- Per-class stats -----
                for i in range(len(labels)):
                    label_i = labels[i].item()
                    if preds[i] == label_i:
                        class_correct[label_i] += 1
                    class_total[label_i] += 1

            # -------------- Scheduler Step --------------
            if phase == 'train':
                scheduler.step()

            # -------------- Compute Averages --------------
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Print to console
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # -------------- Write Summary to Log --------------
            with open(log_file_path, "a") as f:
                f.write(f"{epoch},{phase},{epoch_loss:.4f},{epoch_acc:.4f}\n")

                # Also log per-class accuracies
                for i, cls_name in enumerate(class_names):
                    if class_total[i] > 0:
                        cls_acc = class_correct[i] / class_total[i]
                    else:
                        cls_acc = 0.0
                    f.write(f"   Class '{cls_name}' Accuracy: {cls_acc:.4f}\n")
                f.write("\n")

        print()

    print("Training complete.")

    # 10. Save the model
    exporter = ModelExporter(model, class_names)
    save_path = os.path.join("saved_models", "resnet50_emotion.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    exporter.save(save_path)


if __name__ == "__main__":
    main()