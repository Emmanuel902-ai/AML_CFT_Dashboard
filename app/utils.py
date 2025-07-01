# app/utils.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Class names (example)
CLASS_NAMES = ['benign', 'malignant']

# 2. Preprocessing function
def preprocess_image_for_inference(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0)

# 3. CNN model class (simple example)
class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, num_classes)
        )

    def forward(self, x):
        return self.network(x)
