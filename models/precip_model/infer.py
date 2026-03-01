import os
import json
import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn

# ---------------- CONFIG ----------------
IMAGES_DIR = "../../source_images"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "weather_resnet18_best.pth"

# ---------------- TRANSFORMS ----------------
val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD MODEL ----------------
model = models.resnet18(pretrained=False)  # pretrained not needed for inference

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 3),
    nn.Sigmoid()
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------- LOAD IMAGES ----------------
all_images = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"Running inference on {len(all_images)} images...")

# ---------------- INFERENCE ----------------
results = {}

with torch.no_grad():
    for img_name in all_images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = val_tf(img).unsqueeze(0).to(DEVICE)  # add batch dim

        pred = model(img)
        pred = pred.cpu().numpy().flatten()  # fog, rain, snow

        results[img_name] = {
            "fog_severity": float(pred[0]),
            "rain_severity": float(pred[1]),
            "snow_severity": float(pred[2])
        }

# ---------------- SAVE RESULTS ----------------
import json
with open("predictions.json", "w") as f:
    json.dump(results, f, indent=4)

print("Inference complete. Results saved to predictions.json")