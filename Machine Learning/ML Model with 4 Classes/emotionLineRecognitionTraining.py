import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt

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
# 3. MAIN FUNCTION: Data transforms, dataset loading, model training, saving, plotting,
#    and saving final training metrics in text.
# -----------------------------------------------------------------------------
def main():
    # 0. ENABLE CUDNN BENCHMARK FOR FASTER TRAINING
    cudnn.benchmark = True

    # 1. DATA TRANSFORMS
    data_transforms = transforms.Compose([
        transforms.RandomRotation(15),             # Rotate up to ±15 degrees
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                   # Shift up to 10% in x and y
            scale=(0.8, 1.2),                       # Scale between 80% and 120%
            shear=10                                # Shear angle
        ),
        transforms.RandomHorizontalFlip(),           # Random horizontal flip
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. DATASET LOADING (USING THE ENTIRE DATASET FOR TRAINING)
    custom_classes = ['Angry', 'Anxiety', 'Excitement', 'Sadness']
    data_dir = 'data'  # This folder should contain subfolders for each class.
    dataset = CustomDataset(data_dir, custom_classes, transform=data_transforms)

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("Classes detected:", custom_classes)
    print("Total images:", len(dataset))

    # 3. LOAD & MODIFY RESNET50
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(custom_classes))

    # 4. PREPARE FOR TRAINING
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 100  # Change this value for more epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    use_mixed_precision = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    # Lists to record loss and overall accuracy history per epoch.
    loss_history = []
    accuracy_history = []
    # Dictionary to record per-class accuracy history.
    class_accuracy_history = {class_name: [] for class_name in custom_classes}

    # 5. TRAINING LOOP (ONLY TRAINING PHASE) WITH PER-CLASS ACCURACY
    # In each epoch we will track the number of correct predictions and total items per class.
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Initialize per-class counters.
        num_classes = len(custom_classes)
        class_correct = [0] * num_classes  # Correct predictions per class
        class_total = [0] * num_classes    # Total samples per class

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

            # Update per-class counters.
            for i in range(num_classes):
                # Create a mask for samples belonging to class 'i'
                class_mask = (labels == i)
                class_total[i] += class_mask.sum().item()
                if class_mask.sum().item() > 0:
                    class_correct[i] += (preds[class_mask] == labels[class_mask]).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_acc.item())

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Compute and print per-class accuracies; also record for plotting.
        for idx, class_name in enumerate(custom_classes):
            if class_total[idx] > 0:
                acc = class_correct[idx] / class_total[idx]
                print(f'Class: {class_name:10s} - Acc: {acc:.4f} (Predicted: {class_correct[idx]}/{class_total[idx]})')
                class_accuracy_history[class_name].append(acc)
            else:
                print(f'Class: {class_name:10s} - No samples available.')
                class_accuracy_history[class_name].append(0.0)

        print("\n")

    print('Training complete')

    # -------------------------------------------------------------------------
    # 4. Compute Confusion Matrix for the 4 Emotion Classes
    # -------------------------------------------------------------------------
    from sklearn.metrics import confusion_matrix
    model.eval()
    all_preds = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Capture the final per-class totals and correct prediction counts from the last epoch.
    final_class_total = class_total
    final_class_correct = class_correct

    # Create the directories if they don't exist.
    saved_models_dir = "saved_models"
    results_dir = "results"
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 6. SAVE THE TRAINED MODEL IN THE SAVED_MODELS FOLDER
    exporter = ModelExporter(model, custom_classes)
    save_path = os.path.join(saved_models_dir, "resnet18_custom.pth")
    exporter.save(save_path)

    # 7. SAVE FINAL METRICS TO TEXT FILE (INCLUDING FINAL EPOCH NUMBER AS epoch+1)
    final_epoch_display = num_epochs  # Because epochs are zero-indexed; e.g., for 100 epochs, display "100"
    final_overall_accuracy = accuracy_history[-1]
    final_training_loss = loss_history[-1]
    # Create a dictionary for the final per-class accuracy (from the last recorded epoch).
    final_class_accuracy = {class_name: class_accuracy_history[class_name][-1] for class_name in custom_classes}

    # Build the metrics text including the count details.
    metrics_text = (
        f"Final Training Metrics (Epoch {final_epoch_display}):\n"
        f"Final Training Loss: {final_training_loss:.4f}\n"
        f"Final Overall Accuracy: {final_overall_accuracy:.4f}\n"
        "Per-Class Accuracy and Counts:\n"
    )
    for idx, class_name in enumerate(custom_classes):
        metrics_text += (
            f"  {class_name} - Accuracy: {final_class_accuracy[class_name]:.4f} "
            f"(Predicted Correctly: {final_class_correct[idx]}/{final_class_total[idx]} items)\n"
        )

    # Print final training metrics to the console.
    print(metrics_text)

    # Save the metrics to a text file for later reference.
    metrics_save_path = os.path.join(results_dir, "training_metrics.txt")
    with open(metrics_save_path, "w") as file:
        file.write(metrics_text)
    print(f"Training metrics saved to {metrics_save_path}")

    # -------------------------------------------------------------------------
    # 5. PLOT SEPARATE GRAPHS FOR TRAINING LOSS AND OVERALL ACCURACY
    # -------------------------------------------------------------------------
    epochs = range(1, num_epochs + 1)
    
    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, marker='o', color='red', label='Loss')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot Overall Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy_history, marker='o', color='blue', label='Overall Accuracy')
    plt.title("Overall Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # 9. PLOT CONFUSION MATRIX FOR 4 EMOTION CLASSES WITH PERCENTAGES
    plt.figure(figsize=(10, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for 4 Emotion Classes")
    plt.colorbar()
    tick_marks = range(len(custom_classes))
    plt.xticks(tick_marks, custom_classes, rotation=45)
    plt.yticks(tick_marks, custom_classes)

    # Calculate percentages per row (each row sums to 100%)
    cm_sum = cm.sum(axis=1)[:, None]  # Reshape for broadcasting
    cm_perc = (cm / cm_sum.astype(float)) * 100

    # Annotate cells with percentage values
    for i in range(cm.shape[0]):
       for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_perc[i, j]:.2f}%", ha="center", va="center", 
                     color="white" if cm[i, j] > cm.max()/2. else "black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


if __name__ == '__main__':
    main()