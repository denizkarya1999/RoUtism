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
DATA_ROOT     = "data"  # Root directory with 'train' & 'val' subfolders
CLASSES       = ["Angry", "Anxiety", "Excitement", "Sadness"]
NUM_EPOCHS    = 4      # Total number of training epochs
BATCH_SIZE    = 32       # DataLoader batch size
LEARNING_RATE = 1e-4     # Learning rate for optimizer
WEIGHT_DECAY  = 1e-4     # Weight decay for optimizer
USE_MIXED     = True     # Mixed-precision training flag
TSNE_SAMPLES  = 2048     # Max samples for t-SNE embedding

# ======================= Model Exporter =======================
class ModelExporter:
    """
    Utility to save model weights and class names to a .pth file.
    """
    def __init__(self, model: nn.Module, class_names: list[str]):
        self.model = model
        self.class_names = class_names

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names
        }, path)
        print(f"Model saved to {path}\n")

# ======================= Custom Dataset =======================
class CustomDataset(Dataset):
    """
    Load images from class-named subfolders under a given directory.
    Directory structure:
        phase_dir/<class_name>/*.png|jpg|jpeg
    """
    def __init__(self, phase_dir: str, class_names: list[str], transform=None):
        self.paths, self.labels = [], []
        self.transform = transform
        for idx, cls in enumerate(class_names):
            folder = os.path.join(phase_dir, cls)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(folder, fname))
                    self.labels.append(idx)
        if not self.paths:
            raise RuntimeError(f"No images found in {phase_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ======================= Helpers =======================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, running_correct = 0.0, 0
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
        running_correct += (out.argmax(dim=1) == y).sum().item()
    return running_loss / len(loader.dataset), running_correct

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    correct = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=USE_MIXED):
            out = model(X)
        correct += (out.argmax(dim=1) == y).sum().item()
    return correct

@torch.no_grad()
def compute_confusion(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for X, y in loader:
        X = X.to(device)
        with torch.cuda.amp.autocast(enabled=USE_MIXED):
            out = model(X)
        y_pred += out.argmax(dim=1).cpu().tolist()
        y_true += y.tolist()
    return confusion_matrix(y_true, y_pred)

# ======================= t-SNE + Plot Helpers =======================
def compute_tsne_kmeans(features, n_clusters=len(CLASSES)):
    emb = TSNE(n_components=2, random_state=42).fit_transform(features)
    km  = KMeans(n_clusters=n_clusters, random_state=42).fit(emb)
    return emb, km.cluster_centers_, km.labels_

def plot_metrics(loss_hist, train_acc_hist, val_acc_hist):
    epochs = range(1, len(loss_hist) + 1)
    # Training Loss
    plt.figure()
    plt.plot(epochs, loss_hist, 'o-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    # Train & Val Accuracy
    plt.figure()
    plt.plot(epochs, train_acc_hist, 'o-', label='Train Acc')
    plt.plot(epochs, val_acc_hist,   'o-', label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    # Avg Accuracy
    avg_acc = [(t + v) / 2 for t, v in zip(train_acc_hist, val_acc_hist)]
    plt.figure()
    plt.plot(epochs, avg_acc, 'o-', label='Avg Acc')
    plt.title('Average Accuracy (Train & Val)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_confusion(cm):
    plt.figure()
    cm_sum = cm.sum(axis=1)[:, None]
    cm_perc = cm / cm_sum.astype(float) * 100
    plt.imshow(cm_perc, cmap=plt.cm.Blues, interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('Percent (%)')
    ticks = range(len(CLASSES))
    plt.xticks(ticks, CLASSES, rotation=45)
    plt.yticks(ticks, CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, f"{cm_perc[i,j]:.1f}%", ha='center', va='center',
                     color='white' if cm_perc[i,j] > 50 else 'black')
    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_tsne(emb, centers, labels):
    plt.figure()
    plt.scatter(emb[:,0], emb[:,1], c=labels, s=5, cmap='tab10')
    plt.scatter(centers[:,0], centers[:,1], marker='x', c='k', s=100)
    plt.title('t-SNE + KMeans Clustering')
    plt.show()

# ======================= Main Function =======================
def main():
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_tf = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.8,1.2), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = CustomDataset(os.path.join(DATA_ROOT,'train'), CLASSES, train_tf)
    val_ds   = CustomDataset(os.path.join(DATA_ROOT,'val'),   CLASSES, val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_MIXED)
    loss_hist, train_acc_hist, val_acc_hist = [], [], []
    num_cls = len(CLASSES)
    final_train_corr, final_train_tot = [0]*num_cls, [0]*num_cls
    final_val_corr,   final_val_tot   = [0]*num_cls, [0]*num_cls
    for ep in range(1, NUM_EPOCHS+1):
        train_cls_corr, train_cls_tot = [0]*num_cls, [0]*num_cls
        val_cls_corr,   val_cls_tot   = [0]*num_cls, [0]*num_cls
        loss, train_corr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        for X, y in train_loader:
            preds = model(X.to(device)).argmax(1).cpu()
            for i in range(num_cls):
                mask = (y == i)
                train_cls_tot[i]  += mask.sum().item()
                train_cls_corr[i] += (preds[mask] == i).sum().item()
        val_corr = validate_one_epoch(model, val_loader, device)
        for X, y in val_loader:
            preds = model(X.to(device)).argmax(1).cpu()
            for i in range(num_cls):
                mask = (y == i)
                val_cls_tot[i]  += mask.sum().item()
                val_cls_corr[i] += (preds[mask] == i).sum().item()
        scheduler.step()
        loss_hist.append(loss)
        train_acc_hist.append(train_corr / len(train_ds))
        val_acc_hist.append(val_corr / len(val_ds))
        print(f"Epoch {ep}/{NUM_EPOCHS}  Loss {loss:.4f}  "
              f"Train {train_corr}/{len(train_ds)} ({train_corr/len(train_ds):.2%})  "
              f"Val {val_corr}/{len(val_ds)} ({val_corr/len(val_ds):.2%})")
        # per-class training lines
        print("Per-Class Training Accuracies:")
        for i, cls in enumerate(CLASSES):
            tc, tt = train_cls_corr[i], train_cls_tot[i]
            print(f"  {cls}: {tc}/{tt} ({tc/tt:.2%})")
        # per-class validation lines
        print("Per-Class Validation Accuracies:")
        for i, cls in enumerate(CLASSES):
            vc, vt = val_cls_corr[i], val_cls_tot[i]
            print(f"  {cls}: {vc}/{vt} ({vc/vt:.2%})")
        if ep == NUM_EPOCHS:
            final_train_corr, final_train_tot = train_cls_corr, train_cls_tot
            final_val_corr,   final_val_tot   = val_cls_corr,   val_cls_tot
    cm = compute_confusion(model, val_loader, device)
    plot_confusion(cm)
    feats = []
    with torch.no_grad():
        extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).to(device)
        for X, _ in val_loader:
            feats.append(extractor(X.to(device)).cpu())
    feats = torch.cat(feats,0).numpy()
    emb, centers, labels = compute_tsne_kmeans(feats)
    plot_tsne(emb, centers, labels)
    plot_metrics(loss_hist, train_acc_hist, val_acc_hist)
    os.makedirs('saved_models', exist_ok=True)
    ModelExporter(model, CLASSES).save('saved_models/resnet50_emotion.pth')
    os.makedirs('results', exist_ok=True)
    metrics = (f"Final Epoch: {NUM_EPOCHS}\n"
               f"Final Loss: {loss_hist[-1]:.4f}\n"
               f"Final Overall Train Acc: {train_acc_hist[-1]:.4f} ({train_acc_hist[-1]*100:.2f}%)\n"
               f"Final Overall Val   Acc: {val_acc_hist[-1]:.4f} ({val_acc_hist[-1]*100:.2f}%)\n"
               "Per-Class Train Acc & Counts:\n")
    for i, cls in enumerate(CLASSES):
        tc, tt = final_train_corr[i], final_train_tot[i]
        ta = tc/tt if tt else 0
        metrics += f"  {cls}: {tc}/{tt} ({ta:.4f}) ({ta*100:.2f}%)\n"
    metrics += "Per-Class Val Acc & Counts:\n"
    for i, cls in enumerate(CLASSES):
        vc, vt = final_val_corr[i], final_val_tot[i]
        va = vc/vt if vt else 0
        metrics += f"  {cls}: {vc}/{vt} ({va:.4f}) ({va*100:.2f}%)\n"
    with open('results/training_metrics.txt','w',encoding='utf-8') as f:
        f.write(metrics)

if __name__ == '__main__':
    main()