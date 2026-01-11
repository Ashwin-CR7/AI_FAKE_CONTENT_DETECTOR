import torch
from torchvision import transforms, models
from PIL import Image
import os

CLASS_NAMES = ["ai", "real"]

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model/ai_detector.pth", map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image):
    model = load_model()
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return CLASS_NAMES[pred.item()]
