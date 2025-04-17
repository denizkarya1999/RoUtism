import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ======================= Configuration =======================
# Paths, hyperparameters, and flags
DATA_ROOT     = "data"                     # Root directory for 'train' and 'val' subfolders
CLASSES       = ["Angry", "Anxiety", "Excitement", "Sadness"]  # Emotion categories
NUM_EPOCHS    = 100                        # Number of training epochs
BATCH_SIZE    = 32                         # Batch size for DataLoader
LEARNING_RATE = 1e-4                       # Initial learning rate for optimizer
WEIGHT_DECAY  = 1e-4                       # L2 regularization coefficient
USE_MIXED     = True                       # Enable mixed-precision training
TSNE_SAMPLES  = 2048                       # Max number of feature vectors for t-SNE

# ======================= Model Exporter =======================
class ModelExporter:
    """
    Utility to save a trained PyTorch model and its class names.
    """
    def __init__(self, model: nn.Module, class_names: list[str]):
        self.model = model
        self.class_names = class_names

    def save(self, path: str):
        """
        Save model state and class names to a .pth checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}\n")

# ======================= Custom Dataset =======================
class CustomDataset(Dataset):
    """
    PyTorch Dataset for loading images organized in class-named subfolders.
    Directory structure must be: phase_dir/<class_name>/*.jpg
    """
    def __init__(self, phase_dir: str, class_names: list[str], transform=None):
        self.paths: list[str] = []
        self.labels: list[int] = []
        self.transform = transform
        # Iterate through each class folder and collect file paths and labels
        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(phase_dir, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing folder: {cls_dir}")
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)
        if not self.paths:
            raise RuntimeError(f"No images found in {phase_dir}")

    def __len__(self) -> int:
        # Total number of samples
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Load image and apply transform if provided
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ======================= Helper Functions =======================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Train the model for a single epoch.
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_MIXED):
            out = model(X)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * X.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def validate(model, loader, device):
    """
    Evaluate the model on the validation set.
    Returns:
        accuracy
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=USE_MIXED):
                out = model(X)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
    return correct / len(loader.dataset)


def compute_confusion(model, loader, device):
    """
    Generate confusion matrix for predictions on loader.
    Returns:
        cm: numpy array [num_classes x num_classes]
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            with torch.cuda.amp.autocast(enabled=USE_MIXED):
                out = model(X)
            preds = out.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())
    return confusion_matrix(y_true, y_pred)


def extract_features(model, loader, device, max_samples=TSNE_SAMPLES):
    """
    Extract penultimate-layer features for all samples, then randomly
    sample up to max_samples for t-SNE.
    Returns:
        features: np.ndarray [num_samples x feature_dim]
    """
    # Define extractor by removing final classification layer
    extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).to(device)
    feats = []
    with torch.no_grad():
        for X, _ in loader:
            f = extractor(X.to(device)).cpu()
            feats.append(f)
    feats = torch.cat(feats, dim=0).numpy()
    idxs = np.random.choice(len(feats), min(len(feats), max_samples), replace=False)
    return feats[idxs]


def compute_tsne_kmeans(features, n_clusters=len(CLASSES)):
    """
    Run t-SNE to embed features into 2D, then KMeans cluster.
    Returns:
        emb: np.ndarray [n_samples x 2]
        centers: np.ndarray [n_clusters x 2]
        labels: np.ndarray [n_samples]
    """
    emb = TSNE(n_components=2, random_state=42).fit_transform(features)
    km  = KMeans(n_clusters=n_clusters, random_state=42).fit(emb)
    return emb, km.cluster_centers_, km.labels_


def plot_metrics(loss_hist, train_acc_hist, val_acc_hist):
    """
    Plot training loss and 
    training & validation accuracy over epochs.
    """
    epochs = range(1, len(loss_hist)+1)
    plt.figure()
    plt.plot(epochs, loss_hist, 'o-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_acc_hist, 'o-', label='Train Acc')
    plt.plot(epochs, val_acc_hist,   'o-', label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def plot_confusion(cm):
    """
    Display confusion matrix heatmap with counts.
    """
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    ticks = range(len(CLASSES))
    plt.xticks(ticks, CLASSES, rotation=45)
    plt.yticks(ticks, CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, f"{cm[i,j]}", ha='center', va='center',
                     color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_tsne(emb, centers, labels):
    """
    Scatter plot of 2D t-SNE embedding colored by KMeans labels,
    with cluster centers marked.
    """
    plt.figure()
    plt.scatter(emb[:,0], emb[:,1], c=labels, s=5, cmap='tab10')
    plt.scatter(centers[:,0], centers[:,1], marker='x', c='k', s=100)
    plt.title('t-SNE + KMeans Clustering')
    plt.show()

# ======================= Main Function =======================
def main():
    # Enable CUDNN autotuning for performance
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------ Data Preparation ------------------
    train_tf = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.8,1.2), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = CustomDataset(os.path.join(DATA_ROOT, 'train'), CLASSES, train_tf)
    val_ds   = CustomDataset(os.path.join(DATA_ROOT, 'val'),   CLASSES, val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ------------------ Model Setup ------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED)

    # ------------------ Training Loop ------------------
    loss_hist, train_acc_hist, val_acc_hist = [], [], []
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_acc = validate(model, val_loader, device)
        scheduler.step()

        loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}  "
            f"Loss {train_loss:.4f}  "
            f"TrainAcc {train_acc:.4f}  ValAcc {val_acc:.4f}"
        )

    # ------------------ Evaluation ------------------
    cm = compute_confusion(model, val_loader, device)
    features = extract_features(model, val_loader, device)
    emb, centers, labels = compute_tsne_kmeans(features)

    # ------------------ Visualization ------------------
    plot_metrics(loss_hist, train_acc_hist, val_acc_hist)
    plot_confusion(cm)
    plot_tsne(emb, centers, labels)

    # ------------------ Save Artifacts ------------------
    os.makedirs('saved_models', exist_ok=True)
    ModelExporter(model, CLASSES).save('saved_models/resnet50_emotion.pth')

    os.makedirs('results', exist_ok=True)
    metrics = (
        f"Final Loss: {loss_hist[-1]:.4f}\n"
        f"Final Train Acc: {train_acc_hist[-1]:.4f}\n"
        f"Final Val Acc: {val_acc_hist[-1]:.4f}\n"
    )
    with open('results/training_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(metrics)

# Entry point
if __name__ == '__main__':
    main()
