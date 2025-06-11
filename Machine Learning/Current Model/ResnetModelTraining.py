import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

# FocalLoss to focus on hard examples and mitigate class collapse
def focal_loss(inputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    ce = nn.functional.cross_entropy(inputs, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    fl = ((1 - pt) ** gamma) * ce
    if reduction == 'mean':
        return fl.mean()
    elif reduction == 'sum':
        return fl.sum()
    return fl

class ModelExporter:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def save(self, file_path):
        checkpoint = {'model_state_dict': self.model.state_dict(), 'class_names': self.class_names}
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")


def main():
    torch.backends.cudnn.benchmark = True

    # 1. Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

    # 2. Datasets & Loaders
    data_dir = 'three_class_dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train','val']}
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True,
                             num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True),
        'val':   DataLoader(image_datasets['val'],   batch_size=32, shuffle=False,
                             num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
    class_names = image_datasets['train'].classes

    print("Classes detected:", class_names)
    print("Train samples:", dataset_sizes['train'], "Val samples:", dataset_sizes['val'])

    # 3. Compute alpha for FocalLoss (balanced)
    train_labels = [label for _, label in image_datasets['train'].samples]
    counts = torch.tensor([train_labels.count(i) for i in range(len(class_names))], dtype=torch.float)
    total = counts.sum()
    alpha = total / (len(counts) * counts)
    alpha = alpha.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 4. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    for p in model.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    # 5. Optimizer, scheduler, AMP, loss
    criterion = lambda inputs, targets: focal_loss(inputs, targets, alpha=alpha, gamma=2.0)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

    # Set epochs to 100 and adjust OneCycleLR total_steps accordingly
    num_epochs = 200
    steps_per_epoch = len(dataloaders['train'])
    total_steps = steps_per_epoch * num_epochs
    scheduler = OneCycleLR(optimizer, max_lr=0.02, total_steps=total_steps, anneal_strategy='cos')

    scaler = GradScaler()

    # 6. Training loop
    log_path = "training_log.txt"
    with open(log_path, 'w') as f:
        f.write("epoch,phase,loss,accuracy\n")
    save_path = "saved_models/resnet50_emotion.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    exporter = ModelExporter(model, class_names)

    step_count = 0
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        for phase in ['train','val']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()
            running_loss = 0.0
            running_corrects = 0
            class_correct = [0] * len(class_names)
            class_total = [0] * len(class_names)

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if is_train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    step_count += 1

                running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(labels).sum().item()
                for t, p in zip(labels, preds):
                    class_total[t.item()] += 1
                    class_correct[t.item()] += (p == t).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            with open(log_path, 'a') as f:
                f.write(f"{epoch},{phase},{epoch_loss:.4f},{epoch_acc:.4f}\n")
                for i, cls in enumerate(class_names):
                    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                    f.write(f"   {cls},{phase},acc,{acc:.4f}\n")
                f.write("\n")

        exporter.save(save_path)
    print("Training complete.")

if __name__ == '__main__':
    main()