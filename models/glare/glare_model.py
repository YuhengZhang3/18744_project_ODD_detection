import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from torch.optim import AdamW
from tqdm import tqdm

# --- 1. Custom Dataset Loader ---
class GlareDataset(Dataset):
    def __init__(self, root_dir="glaredetection/RGB", img_size=(512, 512)):
        self.img_paths = sorted(glob.glob(os.path.join(root_dir, "try_*", "images", "*.jpg")))
        self.img_size = img_size
        print(f"Loaded {len(self.img_paths)} image-mask pairs for training.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = img_path.replace("images", "masks").replace(".jpg", "_GT.jpg")

        # Load Image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize image to [0, 1] and channel-first format (C, H, W)
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        # Load Mask
        gt_bgr = cv2.imread(mask_path)
        gt_resized = cv2.resize(gt_bgr, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Parse Red and White as Glare (Class 1)
        gt_white = cv2.inRange(gt_resized, np.array([200, 200, 200]), np.array([255, 255, 255]))
        gt_red = cv2.inRange(gt_resized, np.array([0, 0, 200]), np.array([50, 50, 255]))
        gt_mask = cv2.bitwise_or(gt_white, gt_red)

        # Convert to binary labels (0 = Background, 1 = Glare)
        binary_mask = (gt_mask == 255).astype(np.int64)
        mask_tensor = torch.tensor(binary_mask, dtype=torch.long)

        return img_tensor, mask_tensor

# --- 2. Training Loop ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {str(device).upper()}...")

    # Load Dataset
    dataset = GlareDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Load a pre-trained tiny SegFormer (b0) and change the head to 2 classes
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", # The tiny, lightning-fast base model
        num_labels=2, 
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimizer and Loss function
    optimizer = AdamW(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 15
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Progress bar
        loop = tqdm(dataloader, leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            
            # SegFormer outputs logits at 1/4 size, we must interpolate up to 512x512 to match our mask
            logits = torch.nn.functional.interpolate(
                outputs.logits, 
                size=masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            # Calculate Cross Entropy Loss
            loss = loss_fn(logits, masks)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

    # Save the custom trained weights
    os.makedirs("custom_glare_model", exist_ok=True)
    model.save_pretrained("custom_glare_model")
    print("\nTraining Complete! Model saved to 'custom_glare_model/'.")

if __name__ == "__main__":
    train_model()