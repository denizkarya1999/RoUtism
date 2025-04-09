import torch
import torch.onnx
import torch.nn as nn
import torchvision.models as models

# ========================
# File Paths
# ========================
pth_path = "resnet18_custom.pth"
onnx_path = "resnet18_custom.onnx"

# ========================
# 1. Load the Checkpoint
# ========================
checkpoint = torch.load(pth_path, map_location='cpu')
# Read the class names from the checkpoint; adjust this if your checkpoint uses a different key.
class_names = checkpoint.get('class_names', None)
if class_names is None:
    raise ValueError("Checkpoint does not contain 'class_names'.")
num_classes = len(class_names)
print(f"Number of classes in checkpoint: {num_classes}")

# ========================
# 2. Instantiate the ResNet18 Model Architecture
# ========================
# Create a resnet18 instance. pretrained=False since we're loading our own weights.
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
# Set the final fully connected layer to output the correct number of classes
model.fc = nn.Linear(num_ftrs, num_classes)

# ========================
# 3. Load the State Dictionary
# ========================
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ========================
# 4. Create Dummy Input and Export to ONNX
# ========================
# ResNet18 expects 3-channel images of size 224x224
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

print(f"ONNX model saved to: {onnx_path}")
