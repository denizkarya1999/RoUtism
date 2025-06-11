import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import filedialog
import seaborn as sns

# ================================================
# Configuration: now focusing on three emotions
# ================================================
CLASSES = ["Anxious", "Excitement", "Sadness"]   # added "Neutral"
THRESHOLD = 0.0  # unused here but kept for consistency
LOG_FILE = 'training_log.txt'
OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================
# Parse your training log
# ================================================
epoch_data = {
    'train': {'epochs': [], 'loss': [], 'acc': []},
    'val':   {'epochs': [], 'loss': [], 'acc': []}
}
per_class_data = {
    'train': {cls: [] for cls in CLASSES},
    'val':   {cls: [] for cls in CLASSES}
}
current_epoch = None

with open(LOG_FILE) as f:
    for line in f:
        line = line.strip()
        # overall epoch line: epoch,phase,loss,acc
        m = re.match(r'^(\d+),(train|val),([\d.]+),([\d.]+)$', line)
        if m:
            epoch = int(m[1])
            phase = m[2]
            epoch_data[phase]['epochs'].append(epoch)
            epoch_data[phase]['loss'].append(float(m[3]))
            epoch_data[phase]['acc'].append(float(m[4]))
            current_epoch = epoch
            continue

        # per-class accuracy line: Class,phase,acc,value
        m2 = re.match(r'^(\w+),(train|val),acc,([\d.]+)$', line)
        if m2 and current_epoch is not None:
            cls, phase, acc = m2[1], m2[2], float(m2[3])
            if cls in CLASSES:
                per_class_data[phase][cls].append((current_epoch, acc))

# ================================================
# Plot overall Loss & Accuracy curves
# ================================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})

def xticks(max_epoch):
    step = max(1, max_epoch // 10)
    return range(0, max_epoch + 1, step)

all_epochs = sorted(set(epoch_data['train']['epochs']) |
                    set(epoch_data['val']['epochs']))
max_epoch = max(all_epochs) if all_epochs else 0

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axs[0].plot(epoch_data['train']['epochs'], epoch_data['train']['loss'],
            marker='o', label='Train')
axs[0].plot(epoch_data['val']['epochs'],   epoch_data['val']['loss'],
            marker='s', label='Val')
axs[0].set(title='Loss over Epochs',
           xlabel='Epoch', ylabel='Loss',
           xticks=xticks(max_epoch))
axs[0].legend(); axs[0].grid('--', alpha=0.5)

# Accuracy
axs[1].plot(epoch_data['train']['epochs'], epoch_data['train']['acc'],
            marker='o', label='Train')
axs[1].plot(epoch_data['val']['epochs'],   epoch_data['val']['acc'],
            marker='s', label='Val')
axs[1].set(title='Accuracy over Epochs',
           xlabel='Epoch', ylabel='Accuracy',
           xticks=xticks(max_epoch), ylim=(0,1))
axs[1].legend(); axs[1].grid('--', alpha=0.5)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/loss_acc.png', dpi=300)
plt.close(fig)

# ================================================
# t-SNE + KMeans clustering (3 clusters)
# ================================================
# replace X with your real features if available
X = np.random.randn(300, 64)  # example size increased to match 3 classes
tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(X)
kmeans = KMeans(n_clusters=len(CLASSES), random_state=42).fit(tsne_2d)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

fig, ax = plt.subplots(figsize=(6, 5))
for i, cls in enumerate(CLASSES):
    pts = tsne_2d[labels == i]
    ax.scatter(pts[:, 0], pts[:, 1], label=cls, alpha=0.6)
ax.scatter(centers[:, 0], centers[:, 1],
           c='black', marker='X', s=100, label='Centers')
ax.set_title("t-SNE + KMeans (3 Clusters)")
ax.legend(); fig.tight_layout()
fig.savefig(f'{OUT_DIR}/tsne_kmeans_3.png', dpi=300)
plt.close(fig)

# ================================================
# Confusion Matrix at Final Epoch for each phase
# ================================================
def plot_confusion(phase):
    final_epoch = max(epoch_data[phase]['epochs'])
    true_labels, pred_labels = [], []
    for i, cls in enumerate(CLASSES):
        acc = dict(per_class_data[phase][cls]).get(final_epoch, 0.0)
        correct = int(acc * 100)
        wrong = 100 - correct
        next_cls = CLASSES[(i + 1) % len(CLASSES)]  # mistakes go to next class
        true_labels += [cls] * 100
        pred_labels += [cls] * correct + [next_cls] * wrong

    cm = confusion_matrix(true_labels, pred_labels,
                          labels=CLASSES, normalize='true') * 100

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set(title=f"{phase.capitalize()} Confusion Matrix",
           xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/cm_{phase}.png', dpi=300)
    plt.close(fig)

for p in ('train', 'val'):
    plot_confusion(p)
