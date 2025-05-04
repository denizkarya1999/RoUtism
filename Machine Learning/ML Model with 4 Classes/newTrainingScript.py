#!/usr/bin/env python3
import os
import math
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

########################################
# 1) The SequenceDataset class
########################################
class SequenceDataset(torch.utils.data.Dataset):
    """
    Expects a directory structure like:
      data_with_augmentation/train/<class_name>/<sequence_folder>/frame_*.jpg
      data_with_augmentation/val/<class_name>/<sequence_folder>/frame_*.jpg

    Each <sequence_folder> has N consecutive frames.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # We'll store (sequence_path, label_index) pairs
        self.samples = []
        self.class_to_idx = {}
        
        # classes = sorted list of subfolders (fear, happy, sad, etc.)
        classes = sorted(d for d in os.listdir(self.root_dir) 
                         if os.path.isdir(os.path.join(self.root_dir, d)))
        for i, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = i
            class_dir = os.path.join(self.root_dir, cls_name)
            # each subfolder in class_dir is a sequence
            for seq_folder in sorted(os.listdir(class_dir)):
                seq_path = os.path.join(class_dir, seq_folder)
                if os.path.isdir(seq_path):
                    # store (sequence_folder_path, class_idx)
                    self.samples.append((seq_path, i))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]
        # load all frames in seq_path
        frame_files = sorted(
            f for f in os.listdir(seq_path) 
            if f.lower().endswith(('png','jpg','jpeg'))
        )
        
        frames = []
        for ff in frame_files:
            img_path = os.path.join(seq_path, ff)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        # frames is a list of [C,H,W] Tensors => shape (T,C,H,W)
        frames_tensor = torch.stack(frames, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return frames_tensor, label_tensor

########################################
# 2) The ResNetLSTM model
########################################
class ResNetLSTM(nn.Module):
    def __init__(self, base_model, hidden_dim, num_classes, num_frames=5):
        """
        base_model: a ResNet (or other CNN) for feature extraction.
        hidden_dim: dimension of LSTM hidden state.
        num_classes: how many emotion classes?
        num_frames: max frames per sequence (if you want to fix it).
        """
        super(ResNetLSTM, self).__init__()
        
        self.num_frames = num_frames
        # Remove the final FC from ResNet
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # LSTM input_size = 2048 for ResNet-50
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        x shape: [B, T, C, H, W]
        We'll process each frame with ResNet, then pass the sequence to LSTM.
        """
        B, T, C, H, W = x.shape
        # => (B*T, C, H, W)
        x = x.view(B*T, C, H, W)
        
        # Feature extraction
        feats = self.base_model(x)       # (B*T, 2048, 1, 1)
        feats = feats.view(B*T, 2048)    # => (B*T, 2048)
        
        # Reshape => (B, T, 2048)
        feats = feats.view(B, T, 2048)
        
        # LSTM => (B, T, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(feats)
        # Use the last hidden state => h_n[-1] shape: (B, hidden_dim)
        final_h = h_n[-1]
        
        # Final FC => (B, num_classes)
        logits = self.fc(final_h)
        return logits

########################################
# 3) The main training function
########################################
def main():
    data_dir = "data_with_augmentation"  # root folder
    batch_size = 4
    num_epochs = 10
    hidden_dim = 256
    
    # Transform for each frame
    frame_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Datasets
    train_dataset = SequenceDataset(os.path.join(data_dir, "train"), transform=frame_transform)
    val_dataset   = SequenceDataset(os.path.join(data_dir, "val"),   transform=frame_transform)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    
    class_names = sorted(train_dataset.class_to_idx.keys())
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Load pretrained ResNet, freeze layers if data is small
    base_resnet = models.resnet50(pretrained=True)
    for param in base_resnet.parameters():
        param.requires_grad = False

    # Create ResNet+LSTM
    model = ResNetLSTM(base_model=base_resnet, hidden_dim=hidden_dim, num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # We'll log to training_log.txt
    log_file_path = "training_log.txt"
    # Overwrite (or create) at start
    with open(log_file_path, "w") as f:
        f.write("Epoch,Phase,Loss,Accuracy\n")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-"*10)
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            # Per-class stats
            class_correct = [0]*num_classes
            class_total   = [0]*num_classes
            
            for seq_batch, labels in loader:
                seq_batch = seq_batch.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(seq_batch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase=="train":
                        loss.backward()
                        optimizer.step()
                
                batch_sz = seq_batch.size(0)
                running_loss += loss.item() * batch_sz
                running_corrects += torch.sum(preds == labels)
                total_samples += batch_sz
                
                # Per-class counters
                for i in range(batch_sz):
                    label_i = labels[i].item()
                    if preds[i] == label_i:
                        class_correct[label_i] += 1
                    class_total[label_i] += 1
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            
            # Write logs
            with open(log_file_path, "a") as f:
                f.write(f"{epoch},{phase},{epoch_loss:.4f},{epoch_acc:.4f}\n")
                # Per-class
                for i, cls_name in enumerate(class_names):
                    if class_total[i] > 0:
                        cls_acc = class_correct[i] / class_total[i]
                    else:
                        cls_acc = 0.0
                    f.write(f"   Class '{cls_name}' Accuracy: {cls_acc:.4f}\n")
                f.write("\n")
            
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                # Optionally save best model:
                # torch.save(model.state_dict(), "best_resnetlstm.pth")
        
        print()
    
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    
    # 5) Save final model AND lines info
    # We'll store the model's state_dict, class_names, and the lines used in train_dataset
    save_filename = "resnet_lstm_final.pth"
    lines_info = train_dataset.samples  # list of (seq_path, label_idx)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "lines_info": lines_info
    }
    torch.save(checkpoint, save_filename)
    print(f"Saved final model + lines info to '{save_filename}'")


########################################
# 4) Python entry point
########################################
if __name__ == "__main__":
    main()
