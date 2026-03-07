import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerForSemanticSegmentation
from torch.optim import AdamW
from tqdm import tqdm

# --- 1. Custom Dataset Loader (Unchanged) ---
class GlareDataset(Dataset):
    def __init__(self, root_dir="glaredetection/RGB", img_size=(512, 512)):
        self.img_paths = sorted(glob.glob(os.path.join(root_dir, "try_*", "images", "*.jpg")))
        self.img_size = img_size
        print(f"Found {len(self.img_paths)} total image-mask pairs.")

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
    full_dataset = GlareDataset()
    
    # --- ADDED: Train/Validation Split (80% / 20%) ---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Split dataset: {train_size} training images, {val_size} validation images.")

    # Create distinct DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2) # Don't need to shuffle validation data

    # Load a pre-trained tiny SegFormer (b0) and change the head to 2 classes
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", 
        num_labels=2, 
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimizer and Loss function
    optimizer = AdamW(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 30
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        
        loop = tqdm(train_loader, leave=False, desc=f"Epoch [{epoch+1}/{epochs}] Train")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            
            logits = torch.nn.functional.interpolate(
                outputs.logits, 
                size=masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            loss = loss_fn(logits, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval() # Freeze layers like Dropout/BatchNorm
        total_val_loss = 0.0
        
        # Disable gradient calculation to save memory and compute
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                
                logits = torch.nn.functional.interpolate(
                    outputs.logits, 
                    size=masks.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                
                loss = loss_fn(logits, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save the custom trained weights
    os.makedirs("custom_glare_model", exist_ok=True)
    model.save_pretrained("custom_glare_model")
    print("\nTraining Complete! Model saved to 'custom_glare_model/'.")

if __name__ == "__main__":
    train_model()