import sys
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pneumonia_model.pth"))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load image
img = Image.open(image_path)
img = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)

classes = ["NORMAL", "PNEUMONIA"]
print("Prediction:", classes[pred.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")
