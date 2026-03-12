import os
import json
import cv2
import glob
import torch
import torch.nn as nn
from torchvision import transforms, models

def predict_weather(input_dir="../../source_images", json_dir="output_json", model_path="weather_resnet18_best.pth"):
    # 1. Setup directories
    os.makedirs(json_dir, exist_ok=True)
    
    # Safely grab all images (handles different extensions and cases)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at '{model_path}'. Please check the path.")
        return

    # 2. Setup Device and Transforms
    IMG_SIZE = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Weather ResNet on {device.type.upper()}...")

    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Load Model
    model = models.resnet18(weights=None) 
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Running inference on {len(image_paths)} images...")

    # 4. Process each image individually
    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            # Read and format image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply transforms and add batch dimension
            input_tensor = val_tf(img_rgb).unsqueeze(0).to(device)

            # Predict
            pred = model(input_tensor)
            pred = pred.cpu().numpy().flatten()  # fog, rain, snow

            weather_metrics = {
                "fog_severity": round(float(pred[0]), 4),
                "rain_severity": round(float(pred[1]), 4),
                "snow_severity": round(float(pred[2]), 4)
            }

            # 5. Save/Overwrite individual JSON files
            json_filename = f"{basename}.json"
            json_path = os.path.join(json_dir, json_filename)
            
            # Create a fresh dictionary containing only weather data
            output_data = {
                "weather": weather_metrics
            }
            
            # Save outputs
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"Saved predictions to: {json_filename}")

    print("\nDone! All weather predictions saved to individual files")

if __name__ == "__main__":
    predict_weather()