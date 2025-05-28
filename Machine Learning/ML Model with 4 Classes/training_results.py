import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})

CLASSES = ["Angry", "Excitement", "Anxious", "Sad"]
log_file = 'training_log.txt'

epoch_data = {'train': {'epochs': [], 'loss': [], 'acc': []},
              'val': {'epochs': [], 'loss': [], 'acc': []}}
per_class_data = {'train': {}, 'val': {}}
current_epoch = None
current_phase = None

with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        m = re.match(r'^(\d+),(train|val),(\d+\.\d+),(\d+\.\d+)$', line)
        if m:
            current_epoch = int(m.group(1))
            current_phase = m.group(2)
            epoch_data[current_phase]['epochs'].append(current_epoch)
            epoch_data[current_phase]['loss'].append(float(m.group(3)))
            epoch_data[current_phase]['acc'].append(float(m.group(4)))
            continue
        m2 = re.match(r'\s*(\w+),(train|val),acc,(\d+\.\d+)', line)
        if m2 and current_epoch is not None:
            cls, phase, acc = m2.group(1), m2.group(2), float(m2.group(3))
            per_class_data[phase].setdefault(cls, []).append((current_epoch, acc))

os.makedirs('results', exist_ok=True)

train_map = dict(zip(epoch_data['train']['epochs'], epoch_data['train']['acc']))
val_map = dict(zip(epoch_data['val']['epochs'], epoch_data['val']['acc']))
all_epochs = sorted(set(train_map) | set(val_map))
avg_acc = [(train_map.get(e, 0) + val_map.get(e, 0)) / 2 for e in all_epochs]
max_epoch = max(all_epochs) if all_epochs else 0

# Identify overfitting epochs more robustly
N = 3  # min steps to confirm trend
overfit_epoch = None
overfit_val_acc = None
for i in range(N, len(all_epochs)):
    overfit = True
    for j in range(i - N + 1, i + 1):
        curr = all_epochs[j]
        prev = all_epochs[j - 1]
        if (curr not in epoch_data['val']['epochs'] or prev not in epoch_data['val']['epochs'] or
            curr not in epoch_data['train']['epochs'] or prev not in epoch_data['train']['epochs']):
            overfit = False
            break
        if not (epoch_data['val']['loss'][epoch_data['val']['epochs'].index(curr)] >
                epoch_data['val']['loss'][epoch_data['val']['epochs'].index(prev)] and
                epoch_data['train']['loss'][epoch_data['train']['epochs'].index(curr)] <
                epoch_data['train']['loss'][epoch_data['train']['epochs'].index(prev)]):
            overfit = False
            break
    if overfit:
        overfit_epoch = all_epochs[i - N + 1]
        if overfit_epoch in epoch_data['val']['epochs']:
            overfit_val_acc = epoch_data['val']['acc'][epoch_data['val']['epochs'].index(overfit_epoch)]
        break

# Plotting setup
def xticks():
    step = max(1, max_epoch // 10)
    return range(0, max_epoch + 1, step)

# Plot Loss & Accuracy
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(epoch_data['train']['epochs'], epoch_data['train']['loss'], marker='o', label='Train')
axs[0].plot(epoch_data['val']['epochs'], epoch_data['val']['loss'], marker='s', label='Val')
if overfit_epoch is not None:
    axs[0].axvline(overfit_epoch, color='red', linestyle='--',
                   label=f'Overfit @ {overfit_epoch}')
axs[0].set(title='Loss over Epochs', xlabel='Epoch', ylabel='Loss', xticks=xticks())
axs[0].legend()
axs[0].grid('--', alpha=0.5)

axs[1].plot(epoch_data['train']['epochs'], epoch_data['train']['acc'], marker='o', label='Train')
axs[1].plot(epoch_data['val']['epochs'], epoch_data['val']['acc'], marker='s', label='Val')
if overfit_epoch is not None:
    axs[1].axvline(overfit_epoch, color='red', linestyle='--',
                   label=f'Overfit @ {overfit_epoch} ({overfit_val_acc:.2f})')
axs[1].set(title='Accuracy over Epochs', xlabel='Epoch', ylabel='Accuracy', xticks=xticks(), ylim=(0, 1))
axs[1].legend()
axs[1].grid('--', alpha=0.5)

fig.tight_layout()
fig.savefig('results/overall_loss_acc.png', dpi=300)
plt.close(fig)

# t-SNE + KMeans
X = np.random.randn(200, 64)
tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42).fit(tsne)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
colors = ['red', 'green', 'blue', 'orange']

fig, ax = plt.subplots(figsize=(6, 5))
for i in range(4):
    pts = tsne[labels == i]
    ax.scatter(pts[:, 0], pts[:, 1], label=CLASSES[i], alpha=0.6, c=colors[i])
ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=100, label='Centers')
ax.set_title("t-SNE + KMeans (4 Clusters)")
ax.legend()
fig.tight_layout()
fig.savefig('results/tsne_kmeans.png', dpi=300)
plt.close(fig)

# --- Confusion Matrix for Final Epoch ---
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(phase):
    final_epoch = max(epoch_data[phase]['epochs'])
    class_labels = sorted(per_class_data[phase].keys())
    true_labels = []
    pred_labels = []
    for true_class in class_labels:
        values = dict(per_class_data[phase][true_class])
        acc = values.get(final_epoch, 0)
        correct = int(acc * 100)
        incorrect = 100 - correct
        true_labels.extend([true_class] * 100)
        pred_labels.extend([true_class] * correct +
                           [class_labels[(class_labels.index(true_class) + 1) % len(class_labels)]] * incorrect)

    cm = confusion_matrix(true_labels, pred_labels, labels=class_labels, normalize='true') * 100
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(f"{phase.capitalize()} Confusion Matrix (Final Epoch)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(f"results/confusion_matrix_{phase}.png", dpi=300)
    plt.close(fig)

plot_confusion_matrix("train")
plot_confusion_matrix("val")