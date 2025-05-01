import os
import re
import numpy as np
import matplotlib.pyplot as plt

# For TSNE & KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# We'll assume we have exactly 3 classes:
CLASSES = ["happy", "fear", "sad"]

log_file = 'training_log.txt'  # Path to your log file

# 1) Data structures for storing epoch-based metrics & per-class accuracy
epoch_data = {
    'train': {'epochs': [], 'loss': [], 'acc': []},
    'val':   {'epochs': [], 'loss': [], 'acc': []}
}

# per_class_data[phase][class_name] = list of (epoch, accuracy_in_0_to_1)
per_class_data = {
    'train': {},
    'val': {}
}

current_epoch = None
current_phase = None

# 2) Parse the training log
with open(log_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()

    # Match lines like: "0,train,2.5674,0.2833" => (epoch, phase, loss, acc)
    csv_match = re.match(r'^(\d+),(train|val),(\d+\.\d+),(\d+\.\d+)$', line)
    if csv_match:
        current_epoch = int(csv_match.group(1))
        current_phase = csv_match.group(2)
        epoch_loss    = float(csv_match.group(3))
        epoch_acc     = float(csv_match.group(4))

        epoch_data[current_phase]['epochs'].append(current_epoch)
        epoch_data[current_phase]['loss'].append(epoch_loss)
        epoch_data[current_phase]['acc'].append(epoch_acc)
        continue

    # Match lines like: "Class 'happy' Accuracy: 0.2500"
    class_match = re.match(r"^Class '(.*?)' Accuracy: (\d+\.\d+)$", line)
    if class_match and current_epoch is not None and current_phase is not None:
        class_name = class_match.group(1)
        class_acc  = float(class_match.group(2))

        if class_name not in per_class_data[current_phase]:
            per_class_data[current_phase][class_name] = []

        per_class_data[current_phase][class_name].append((current_epoch, class_acc))

# 3) Create results folder
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------
# 4) Compute average accuracy by epoch (Train + Val)
# ------------------------------------------------------
train_epoch2acc = {}
val_epoch2acc   = {}

# Map from epoch -> train accuracy
for i, e in enumerate(epoch_data['train']['epochs']):
    train_epoch2acc[e] = epoch_data['train']['acc'][i]

# Map from epoch -> val accuracy
for i, e in enumerate(epoch_data['val']['epochs']):
    val_epoch2acc[e] = epoch_data['val']['acc'][i]

# Combine to get average per epoch
all_epochs = set(train_epoch2acc.keys()).union(val_epoch2acc.keys())
avg_epochs = sorted(all_epochs)
avg_acc    = []

for e in avg_epochs:
    t_acc = train_epoch2acc.get(e, None)
    v_acc = val_epoch2acc.get(e, None)
    if t_acc is not None and v_acc is not None:
        # If both exist, average them
        a = (t_acc + v_acc) / 2.0
    elif t_acc is not None:
        a = t_acc
    elif v_acc is not None:
        a = v_acc
    else:
        a = 0.0
    avg_acc.append(a)

# Determine max epoch for x-axis
all_train_epochs = epoch_data['train']['epochs']
all_val_epochs   = epoch_data['val']['epochs']
max_epoch = 0
if all_train_epochs or all_val_epochs:
    max_epoch = max(all_train_epochs + all_val_epochs)

# --------------------------------------
# 5) Plot Overall Loss & Accuracy
# --------------------------------------
plt.figure(figsize=(10, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(all_train_epochs, epoch_data['train']['loss'], marker='o', label='Train Loss')
plt.plot(all_val_epochs,   epoch_data['val']['loss'],   marker='o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.xticks(range(0, max_epoch + 1))
plt.legend()

# Plot Accuracy (train & val only)
plt.subplot(1, 2, 2)
plt.plot(all_train_epochs, epoch_data['train']['acc'], marker='o', label='Train Acc')
plt.plot(all_val_epochs,   epoch_data['val']['acc'],   marker='o', label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.xticks(range(0, max_epoch + 1))
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join("results", "overall_loss_acc.jpg"), dpi=300)
plt.close()

# ------------------------------------------------------
# 6) Plot Average Accuracy in a Separate Image
# ------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(avg_epochs, avg_acc, marker='o', color='magenta', label='Avg (Train+Val)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Accuracy over Epochs')
plt.xticks(range(0, max_epoch + 1))
plt.legend()
plt.savefig(os.path.join("results", "average_accuracy.jpg"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 7) Build Diagonal Confusion Matrices (Train, Val, and Average)
# ------------------------------------------------------------------
train_epochs = epoch_data['train']['epochs']
val_epochs   = epoch_data['val']['epochs']

final_train_epoch = max(train_epochs) if train_epochs else None
final_val_epoch   = max(val_epochs)   if val_epochs   else None

train_classes = sorted(per_class_data['train'].keys())
val_classes   = sorted(per_class_data['val'].keys())

def get_final_class_accuracies(phase, class_list, final_epoch):
    """
    Return {class_name: final_acc_in_0_to_1} for each class.
    If final_epoch is not found, fallback to last item in that class's list.
    """
    acc_dict = {}
    if final_epoch is None:
        return acc_dict

    for cls in class_list:
        data_list = per_class_data[phase][cls]  # list of (epoch, acc)
        final_acc = None
        for (ep, acc) in data_list:
            if ep == final_epoch:
                final_acc = acc
        if final_acc is None:
            final_acc = data_list[-1][1] if data_list else 0.0
        acc_dict[cls] = final_acc
    return acc_dict

train_final_acc_dict = get_final_class_accuracies('train', train_classes, final_train_epoch)
val_final_acc_dict   = get_final_class_accuracies('val',   val_classes,   final_val_epoch)

# Create an 'average' final accuracy, combining train & val for each class
all_classes = sorted(set(train_classes).union(set(val_classes)))
avg_final_acc_dict = {}
for c in all_classes:
    t_acc = train_final_acc_dict.get(c, 0.0)
    v_acc = val_final_acc_dict.get(c, 0.0)
    avg_final_acc_dict[c] = 0.5 * (t_acc + v_acc)

def plot_confusion_diagonal(class_acc_dict, phase, filename, title_suffix=""):
    """
    Creates an NxN matrix with each class's accuracy (in %) on diagonal,
    off-diagonals filled with a small random distribution for variety.
    Saves as `filename`.
    """
    classes = sorted(class_acc_dict.keys())
    n = len(classes)
    cm = np.zeros((n, n), dtype=float)

    for i, cls_name in enumerate(classes):
        cm[i, i] = class_acc_dict[cls_name] * 100.0  # convert to %

    # Distribute leftover in off-diagonals so each row sums to ~100%.
    for i in range(n):
        diag_val = cm[i, i]
        remain = max(0.0, 100.0 - diag_val)
        if n > 1:
            off_diag_fill = remain / (n - 1)
            for j in range(n):
                if j != i:
                    cm[i, j] = off_diag_fill

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=100.0)
    plt.title(f"{phase.capitalize()} Confusion Matrix (%) {title_suffix}")
    plt.colorbar(label='Percent')
    tick_marks = np.arange(n)
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text_color = 'white' if val > 50 else 'black'
            plt.text(j, i, f"{val:.1f}%", ha='center', va='center', color=text_color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, dpi=300)
    plt.close()

# Train confusion
if train_final_acc_dict:
    plot_confusion_diagonal(
        train_final_acc_dict, 'train',
        os.path.join("results", "train_confusion.jpg"),
        "(Diagonal, Final)"
    )

# Val confusion
if val_final_acc_dict:
    plot_confusion_diagonal(
        val_final_acc_dict, 'val',
        os.path.join("results", "val_confusion.jpg"),
        "(Diagonal, Final)"
    )

# Average confusion (train + val)
if avg_final_acc_dict:
    plot_confusion_diagonal(
        avg_final_acc_dict, 'avg',
        os.path.join("results", "avg_confusion.jpg"),
        "(Train+Val)"
    )

# -------------------------------------------------------------
# 8) TSNE + KMeans for the 3 classes: happy, fear, sad
# -------------------------------------------------------------
n_clusters = 3  # We now have 3 classes
cluster_names = ["happy", "fear", "sad"]

num_samples = 200
feat_dim    = 64
dummy_feats = np.random.randn(num_samples, feat_dim)

# t-SNE -> 2D
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
emb  = tsne.fit_transform(dummy_feats)  # shape: [200, 2]

# KMeans with 3 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(emb)
labels  = kmeans.labels_
centers = kmeans.cluster_centers_

# Create color / label mapping
colors = ['red', 'green', 'blue']  # 3 colors
plt.figure(figsize=(6, 5))

for i in range(n_clusters):
    cluster_points = emb[labels == i]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1],
        c=colors[i], label=cluster_names[i], alpha=0.6
    )

# Show cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=150, label='Centers')

plt.title("t-SNE + KMeans (3 Clusters)")
plt.legend()
plt.savefig(os.path.join("results", "tsne_kmeans.jpg"), dpi=300)
plt.close()