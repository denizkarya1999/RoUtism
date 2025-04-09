import torch
import torch.nn as nn
from torchvision import transforms, models
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import os

class InferenceEngine:
    def __init__(self, checkpoint_path, device):
        """
        Args:
            checkpoint_path: Path to the saved .pth model checkpoint.
            device: Device to run inference on.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model, self.class_names = self.load_checkpoint()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_checkpoint(self):
        # Load the checkpoint.
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        # Initialize the model architecture (ResNet18) and update its final layer.
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model, class_names

    def predict(self, image_path):
        """Perform inference on the image at image_path and return top 3 probabilities."""
        image = Image.open(image_path)
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)  # create a mini-batch
        
        with torch.no_grad():
            output = self.model(input_batch)
            # Compute probabilities using softmax.
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Get top 3 predictions.
            top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        # Build a dictionary for the top 3 classes with their probabilities.
        results = {}
        for i in range(3):
            class_name = self.class_names[top3_indices[0, i].item()]
            prob = top3_prob[0, i].item()
            results[class_name] = prob
        
        return results

def main():
    # Set device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define the checkpoint path.
    checkpoint_path = os.path.join("saved_models", "resnet18_custom.pth")
    if not os.path.exists(checkpoint_path):
        raise Exception(f"Checkpoint not found at {checkpoint_path}")
    
    # Create an instance of InferenceEngine.
    inference_engine = InferenceEngine(checkpoint_path, device)
    
    # Hide the main tkinter window.
    Tk().withdraw()
    
    # Open file dialog to select an image.
    filename = askopenfilename(
        title="Select an image file for inference",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not filename:
        raise Exception("No image file selected.")
    
    # Predict the top 3 classes with probabilities.
    predictions = inference_engine.predict(filename)
    print("Top 3 Predictions (class: probability):")
    for cls, prob in predictions.items():
        print(f"{cls}: {prob:.4f}")

if __name__ == "__main__":
    main()