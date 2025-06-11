import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import Counter
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

# ================= Configuration =================
CLASSES = ["Anxious", "Excitement", "Sadness"]
THRESHOLD = 0.00  # below this confidence → "Neutral"

# pre‐define transform once for efficiency
TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ================ Helpers =======================
def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = models.resnet50(weights=None)
    # adjust final layer to match our three classes
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model

def infer_folder(model, folder: str, device: torch.device, output_widget):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not files:
        messagebox.showerror("No Images", f"No images found in {folder}")
        return

    counts = Counter()
    total = len(files)
    output_widget.insert(tk.END,
                         f"Inferencing {total} images in '{folder}'\n\n")

    for fname in files:
        path = os.path.join(folder, fname)
        img  = Image.open(path).convert("RGB")
        x    = TF(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out   = model(x)
            probs = torch.softmax(out, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        label = CLASSES[idx] if conf.item() >= THRESHOLD else "Neutral"
        counts[label] += 1
        pct = conf.item() * 100

        output_widget.insert(
            tk.END,
            f"{fname:30s} → {label:10s} ({pct:5.2f}%)\n"
        )

    # summary (Anxious, Excitement, Sadness, Neutral)
    output_widget.insert(tk.END, "\nSummary:\n")
    for cls in CLASSES + ["Neutral"]:
        c = counts[cls]
        output_widget.insert(
            tk.END,
            f"  {cls:10s}: {c:3d}/{total:3d} ({c/total:5.2%})\n"
        )

# ================ GUI ===========================
class InferenceApp:
    def __init__(self, root):
        self.root = root
        root.title("RoEmotion Inference: Anxious, Excitement, Sadness")

        tk.Label(root, text="Model checkpoint:").grid(row=0, column=0,
                                                     sticky="e")
        self.ckpt_path_var = tk.StringVar()
        tk.Entry(root, textvariable=self.ckpt_path_var,
                 width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse…", command=self.browse_ckpt)\
            .grid(row=0, column=2)

        tk.Label(root, text="Images folder:").grid(row=1, column=0,
                                                   sticky="e")
        self.folder_var = tk.StringVar()
        tk.Entry(root, textvariable=self.folder_var,
                 width=50).grid(row=1, column=1)
        tk.Button(root, text="Browse…", command=self.browse_folder)\
            .grid(row=1, column=2)

        tk.Button(root, text="Run Inference",
                  command=self.run_inference)\
            .grid(row=2, column=1, pady=5)

        self.output = scrolledtext.ScrolledText(root, width=80,
                                                height=20)
        self.output.grid(row=3, column=0, columnspan=3,
                         padx=5, pady=5)

    def browse_ckpt(self):
        p = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch Checkpoint","*.pth"),
                       ("All files","*.*")]
        )
        if p:
            self.ckpt_path_var.set(p)

    def browse_folder(self):
        d = filedialog.askdirectory(title="Select folder with images")
        if d:
            self.folder_var.set(d)

    def run_inference(self):
        ckpt = self.ckpt_path_var.get().strip()
        folder = self.folder_var.get().strip()
        if not os.path.isfile(ckpt):
            messagebox.showerror("Error",
                                 "Please select a valid checkpoint file.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Error",
                                 "Please select a valid images folder.")
            return

        self.output.delete("1.0", tk.END)
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        model = load_model(ckpt, device)
        infer_folder(model, folder, device, self.output)


if __name__ == "__main__":
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()