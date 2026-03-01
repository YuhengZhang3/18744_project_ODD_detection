import os
import json
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------

IMAGES_DIR = "images"
JSON_DIR = "json"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

MAX_IMAGES = 10000

BEST_MODEL_PATH = "weather_resnet18_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DEBUG INFO ----------------

print("\n--- DEBUG INFO ---")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    print(
        "Total GPU Memory (GB):",
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    )
    torch.backends.cudnn.benchmark = True
else:
    print("Running on CPU")

print("------------------\n")

# ---------------- DATASET ----------------

class WeatherDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        img_path = os.path.join(IMAGES_DIR, img_name)
        json_path = os.path.join(JSON_DIR, img_name.rsplit(".", 1)[0] + ".json")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        with open(json_path, "r") as f:
            label = json.load(f)

        target = torch.tensor(
            [
                label["fog_severity"],
                label["rain_severity"],
                label["snow_severity"],
            ],
            dtype=torch.float32,
        )

        return img, target

# ---------------- TRANSFORMS ----------------

train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------- LOAD FILES ----------------

all_images = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print("Total available images:", len(all_images))

random.seed(42)
random.shuffle(all_images)
all_images = all_images[:MAX_IMAGES]

print("Using images:", len(all_images))

# ---------------- SPLIT: TRAIN / VAL / TEST ----------------

train_files, temp_files = train_test_split(
    all_images, test_size=0.3, random_state=42
)

val_files, test_files = train_test_split(
    temp_files, test_size=0.5, random_state=42
)

print("Train size:", len(train_files))
print("Val size:", len(val_files))
print("Test size:", len(test_files))

# ---------------- DATALOADERS ----------------

train_ds = WeatherDataset(train_files, train_tf)
val_ds = WeatherDataset(val_files, val_tf)
test_ds = WeatherDataset(test_files, val_tf)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True,
)

# ---------------- MODEL ----------------

print("\nLoading pretrained ResNet18...")

model = models.resnet18(pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 3),
    nn.Sigmoid(),
)

model = model.to(DEVICE)

print("Model loaded.\n")

# ---------------- TRAINING SETUP ----------------

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

# ---------------- TRAINING LOOP ----------------

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, targets in train_loader:
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ---- SAVE BEST MODEL ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  ✓ Saved new best model (val loss = {val_loss:.4f})")

# ---------------- FINAL TEST EVALUATION ----------------

print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

test_loss = 0.0

with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, targets)
        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"\nFinal Test Loss (best model): {test_loss:.4f}")
print(f"Best model saved to: {BEST_MODEL_PATH}")