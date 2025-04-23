import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ======================= Configuration =======================
DATA_ROOT     = "data"   # Root directory with 'train' & 'val' subfolders
CLASSES       = ["Angry", "Anxiety", "Excitement", "Sadness"]
NUM_EPOCHS    = 100       # Total number of training epochs
BATCH_SIZE    = 16       # DataLoader batch size
LEARNING_RATE = 1e-4     # Learning rate for optimizer
WEIGHT_DECAY  = 1e-4     # Weight decay for optimizer
USE_MIXED     = True     # Mixed-precision training flag
TSNE_SAMPLES  = 2880     # Max samples for t-SNE embedding

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
    """
    Run one training epoch; return average loss over dataset, total correct.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0

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

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, running_correct

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """
    Run validation over dataset; return total correct and average loss.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=USE_MIXED):
            out = model(X)
            loss = criterion(out, y)
        running_loss += loss.item() * X.size(0)
        running_correct += (out.argmax(dim=1) == y).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    return running_correct, avg_loss

@torch.no_grad()
def compute_confusion(model, loader, device):
    """
    Return confusion matrix by comparing true vs. predicted labels
    across the given loader.
    """
    model.eval()
    y_true, y_pred = [], []
    for X, y in loader:
        X = X.to(device)
        with torch.cuda.amp.autocast(enabled=USE_MIXED):
            out = model(X)
        y_pred.extend(out.argmax(dim=1).cpu().tolist())
        y_true.extend(y.tolist())
    return confusion_matrix(y_true, y_pred)

# ======================= t-SNE + Plot Helpers =======================
def compute_tsne_kmeans(features, n_clusters=len(CLASSES)):
    emb = TSNE(n_components=2, random_state=42).fit_transform(features)
    km  = KMeans(n_clusters=n_clusters, random_state=42).fit(emb)
    return emb, km.cluster_centers_, km.labels_

def plot_metrics(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist):
    """
    Plot training loss, validation loss, training accuracy, 
    validation accuracy, and average accuracy across epochs.
    """
    epochs = range(1, len(train_loss_hist) + 1)

    # Training & Validation Loss
    plt.figure()
    plt.plot(epochs, train_loss_hist, 'o-', label='Train Loss')
    plt.plot(epochs, val_loss_hist,   'o-', label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Train & Val Accuracy
    plt.figure()
    plt.plot(epochs, train_acc_hist, 'o-', label='Train Acc')
    plt.plot(epochs, val_acc_hist,   'o-', label='Val Acc')
    plt.title('Accuracy Curves')
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
    """
    Display a confusion matrix with percentage formatting.
    """
    plt.figure()
    cm_sum = cm.sum(axis=1)[:, None]
    cm_perc = (cm / cm_sum.astype(float)) * 100
    plt.imshow(cm_perc, cmap=plt.cm.Blues, interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('Percent (%)')

    ticks = range(len(CLASSES))
    plt.xticks(ticks, CLASSES, rotation=45)
    plt.yticks(ticks, CLASSES)

    # Annotate each cell
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
    """
    Plot a t-SNE embedding with KMeans centers.
    """
    plt.figure()
    plt.scatter(emb[:,0], emb[:,1], c=labels, s=5, cmap='tab10')
    plt.scatter(centers[:,0], centers[:,1], marker='x', c='k', s=100)
    plt.title('t-SNE + KMeans Clustering')
    plt.show()

# ======================= Main Function =======================
def main():
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- DATA AUGMENTATION ---
    train_tf = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.8,1.2), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- DATASETS & LOADERS ---
    train_ds = CustomDataset(os.path.join(DATA_ROOT, 'train'), CLASSES, train_tf)
    val_ds   = CustomDataset(os.path.join(DATA_ROOT, 'val'),   CLASSES, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Combined dataset for the "average" confusion matrix
    combined_ds = ConcatDataset([train_ds, val_ds])
    combined_loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True)

    # --- MODEL SETUP ---
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_MIXED)

    # Histories
    train_loss_hist = []
    val_loss_hist   = []
    train_acc_hist  = []
    val_acc_hist    = []

    # For final per-class stats
    num_cls = len(CLASSES)
    final_train_corr, final_train_tot = [0]*num_cls, [0]*num_cls
    final_val_corr,   final_val_tot   = [0]*num_cls, [0]*num_cls

    # --- TRAINING LOOP ---
    for ep in range(1, NUM_EPOCHS+1):
        # --- TRAIN ---
        train_loss, train_corr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Compute per-class stats for training
        train_cls_corr = [0]*num_cls
        train_cls_tot  = [0]*num_cls
        for X, y in train_loader:
            preds = model(X.to(device)).argmax(1).cpu()
            for i in range(num_cls):
                mask = (y == i)
                train_cls_tot[i]  += mask.sum().item()
                train_cls_corr[i] += (preds[mask] == i).sum().item()

        # --- VALIDATE ---
        val_corr, val_loss = validate_one_epoch(model, val_loader, criterion, device)

        # Compute per-class stats for validation
        val_cls_corr = [0]*num_cls
        val_cls_tot  = [0]*num_cls
        for X, y in val_loader:
            preds = model(X.to(device)).argmax(1).cpu()
            for i in range(num_cls):
                mask = (y == i)
                val_cls_tot[i]  += mask.sum().item()
                val_cls_corr[i] += (preds[mask] == i).sum().item()

        # LR schedule step
        scheduler.step()

        # Save history
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_corr / len(train_ds))
        val_acc_hist.append(val_corr / len(val_ds))

        # --- CONSOLE LOGGING ---
        print(f"Epoch {ep}/{NUM_EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
              f"Train Acc: {train_corr}/{len(train_ds)} ({train_corr/len(train_ds):.2%})  "
              f"Val Acc: {val_corr}/{len(val_ds)} ({val_corr/len(val_ds):.2%})\n")

        print("Per-Class Training Accuracies:")
        for i, cls in enumerate(CLASSES):
            tc, tt = train_cls_corr[i], train_cls_tot[i]
            print(f"  {cls}: {tc}/{tt} ({tc/tt:.2%})")

        print("Per-Class Validation Accuracies:")
        for i, cls in enumerate(CLASSES):
            vc, vt = val_cls_corr[i], val_cls_tot[i]
            print(f"  {cls}: {vc}/{vt} ({vc/vt:.2%})")

        print("-"*60)

        # Save final (last epoch) per-class stats
        if ep == NUM_EPOCHS:
            final_train_corr, final_train_tot = train_cls_corr, train_cls_tot
            final_val_corr,   final_val_tot   = val_cls_corr,   val_cls_tot

    # ======== CONFUSION MATRICES ========
    print("\nTraining Confusion Matrix:")
    cm_train = compute_confusion(model, train_loader, device)
    plot_confusion(cm_train)

    print("\nValidation Confusion Matrix:")
    cm_val = compute_confusion(model, val_loader, device)
    plot_confusion(cm_val)

    print("\nAverage (Train+Val) Confusion Matrix:")
    cm_combined = compute_confusion(model, combined_loader, device)
    plot_confusion(cm_combined)

    # ----- t-SNE from validation set -----
    feats = []
    with torch.no_grad():
        extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).to(device)
        for X, _ in val_loader:
            feats.append(extractor(X.to(device)).cpu())
    feats = torch.cat(feats, 0).numpy()
    emb, centers, labels = compute_tsne_kmeans(feats)
    plot_tsne(emb, centers, labels)

    # ----- Plot Metrics -----
    plot_metrics(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist)

    # ----- Save Model -----
    os.makedirs('saved_models', exist_ok=True)
    ModelExporter(model, CLASSES).save('saved_models/resnet50_emotion.pth')

    # ----- Write Stats to File -----
    os.makedirs('results', exist_ok=True)

    final_train_loss = train_loss_hist[-1]
    final_val_loss   = val_loss_hist[-1]
    final_train_acc  = train_acc_hist[-1]
    final_val_acc    = val_acc_hist[-1]

    metrics = (
        f"Final Epoch: {NUM_EPOCHS}\n"
        f"Final Train Loss: {final_train_loss:.4f}\n"
        f"Final Val Loss:   {final_val_loss:.4f}\n"
        f"Final Overall Train Acc: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)\n"
        f"Final Overall Val   Acc: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)\n"
        "Per-Class Train Acc & Counts:\n"
    )

    for i, cls in enumerate(CLASSES):
        tc, tt = final_train_corr[i], final_train_tot[i]
        ta = tc / tt if tt else 0
        metrics += f"  {cls}: {tc}/{tt} ({ta:.4f}) ({ta*100:.2f}%)\n"

    metrics += "Per-Class Val Acc & Counts:\n"
    for i, cls in enumerate(CLASSES):
        vc, vt = final_val_corr[i], final_val_tot[i]
        va = vc / vt if vt else 0
        metrics += f"  {cls}: {vc}/{vt} ({va:.4f}) ({va*100:.2f}%)\n"

    # ----- Per-Class Ave Acc & Counts -----
    metrics += "Per-Class Ave Acc & Counts:\n"
    ave_corr_list = []
    ave_tot_list  = []
    for i, cls in enumerate(CLASSES):
        ave_corr = final_train_corr[i] + final_val_corr[i]
        ave_tot  = final_train_tot[i]  + final_val_tot[i]
        ave_acc  = ave_corr / ave_tot if ave_tot else 0
        ave_corr_list.append(ave_corr)
        ave_tot_list.append(ave_tot)
        metrics += f"  {cls}: {ave_corr}/{ave_tot} ({ave_acc:.4f}) ({ave_acc*100:.2f}%)\n"

    # ----- Final Average Accuracy (Train + Val) -----
    total_ave_corr = sum(ave_corr_list)
    total_ave_tot  = sum(ave_tot_list)
    final_ave_acc = total_ave_corr / total_ave_tot if total_ave_tot else 0
    metrics += (f"Final Average Accuracy: {total_ave_corr}/{total_ave_tot} "
                f"({final_ave_acc:.4f}) ({final_ave_acc*100:.2f}%)\n")

    with open('results/training_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(metrics)

if __name__ == '__main__':
    main()