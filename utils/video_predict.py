import cv2
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

def predict_video(video_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)

    ai_count = 0
    real_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 300:
            break

        frame_count += 1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)

        if CLASS_NAMES[pred.item()] == "ai":
            ai_count += 1
        else:
            real_count += 1

    cap.release()
    return ai_count, real_count
