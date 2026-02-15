print("Starting evaluation...")

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = datasets.ImageFolder("data/chest_xray/test", transform=transform)
print("Test images found:", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=32)

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

print("Loading model...")
model.load_state_dict(torch.load("pneumonia_model.pth"))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        print("Running batch...")
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Finished inference")

if len(all_labels) > 0:
    print(classification_report(all_labels, all_preds))
else:
    print("No test data found.")
