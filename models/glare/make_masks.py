import os
import cv2
import glob
import numpy as np

def create_blank_masks(target_dir="glaredetection/RGB/try_negatives"):
    images_dir = os.path.join(target_dir, "images")
    masks_dir = os.path.join(target_dir, "masks")
    
    # 1. Create the directories if they don't exist yet
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # 2. Find all the sky images you dropped in
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    
    if not image_paths:
        print(f"Folders created at '{target_dir}'!")
        print(f"Please copy your blue sky images into '{images_dir}' and run this script again.")
        return
        
    print(f"Found {len(image_paths)} sky images. Generating blank ground-truth masks...")
    
    count = 0
    for img_path in image_paths:
        # Read the image just to get its exact pixel dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
            
        h, w = img.shape[:2]
        
        # Create a completely pitch-black image (all zeros)
        blank_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Construct the exact mask filename your dataloader expects
        base_name = os.path.basename(img_path)
        mask_name = base_name.replace(".jpg", "_GT.jpg")
        mask_path = os.path.join(masks_dir, mask_name)
        
        # Save it
        cv2.imwrite(mask_path, blank_mask)
        count += 1
        
    print(f"\nSuccess! Created {count} blank masks in '{masks_dir}'.")
    print("You can now run 'python3 glare_model.py' to retrain your model with these hard negatives included.")

if __name__ == "__main__":
    create_blank_masks()