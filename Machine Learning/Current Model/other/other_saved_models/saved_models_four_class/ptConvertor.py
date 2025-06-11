import os
import torch
import torch.nn as nn
from torchvision import models
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

def pick_pth():
    root = Tk(); root.withdraw()
    pth_path = askopenfilename(
        title="Select your .pth checkpoint",
        filetypes=[("PyTorch Checkpoint", "*.pth"), ("All files", "*.*")]
    )
    root.destroy()
    return pth_path

def pick_pt():
    root = Tk(); root.withdraw()
    pt_path = asksaveasfilename(
        title="Save TorchScript model as",
        defaultextension=".pt",
        filetypes=[("TorchScript Model", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return pt_path

def convert():
    pth_path = pick_pth()
    if not pth_path:
        print("❌ No .pth selected—exiting.")
        return

    # 1) load checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    class_names = checkpoint.get("class_names", None)
    if class_names is None:
        raise KeyError("`class_names` not found in checkpoint—was this saved with ModelExporter?")

    # 2) rebuild model with pretrained backbone
    num_classes = len(class_names)
    model = models.resnet50(pretrained=True)      # <-- pretrained=True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) trace
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)

    # 4) save
    pt_path = pick_pt()
    if not pt_path:
        print("❌ No output path selected—exiting.")
        return

    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.jit.save(traced, pt_path)
    print(f"✅ TorchScript model saved to: {pt_path}")

if __name__ == "__main__":
    convert()