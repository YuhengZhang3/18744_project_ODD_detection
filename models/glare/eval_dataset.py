import os
import cv2
import glob
import numpy as np

def run_grid_search(dataset_dir="glaredetection/RGB"):
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "try_*", "images", "*.jpg")))
    
    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images. Starting Hyperparameter Grid Search...\n")
    
    # The thresholds we want to test
    v_thresholds = [200, 215, 230, 240, 250]
    s_thresholds = [20, 40, 60, 80, 100]
    
    results = []

    for v_thresh in v_thresholds:
        for s_thresh in s_thresholds:
            total_iou = 0.0
            valid_images = 0
            
            for img_path in image_paths:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                    
                h, w = img_bgr.shape[:2]

                mask_path = img_path.replace("images", "masks").replace(".jpg", "_GT.jpg")
                gt_bgr = cv2.imread(mask_path)
                if gt_bgr is None: continue

                # Resize GT to match image
                gt_bgr = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

                # Parse GT Mask (Red + White)
                gt_white = cv2.inRange(gt_bgr, np.array([200, 200, 200]), np.array([255, 255, 255]))
                gt_red = cv2.inRange(gt_bgr, np.array([0, 0, 200]), np.array([50, 50, 255]))
                gt_mask = cv2.bitwise_or(gt_white, gt_red)

                # Run Algorithm
                hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                S_channel = hsv_img[:, :, 1]
                V_channel = hsv_img[:, :, 2]
                
                pred_mask = ((V_channel > v_thresh) & (S_channel < s_thresh)).astype(np.uint8) * 255
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

                # Calculate IoU
                intersection = np.logical_and(gt_mask == 255, pred_mask == 255)
                union = np.logical_or(gt_mask == 255, pred_mask == 255)
                
                union_sum = np.sum(union)
                if union_sum == 0:
                    iou = 1.0 
                else:
                    iou = np.sum(intersection) / union_sum
                    
                total_iou += iou
                valid_images += 1
            
            if valid_images > 0:
                mean_iou = total_iou / valid_images
                results.append( (mean_iou, v_thresh, s_thresh) )
                # Print progress so you know it hasn't frozen
                print(f"Tested V>{v_thresh}, S<{s_thresh} -> IoU: {mean_iou * 100:.2f}%")

    # Sort results to find the highest IoU
    results.sort(reverse=True, key=lambda x: x[0])
    
    print("\n=== TOP 3 BEST COMBINATIONS ===")
    for i in range(min(3, len(results))):
        score, v, s = results[i]
        print(f"{i+1}. V > {v}, S < {s}  -->  IoU: {score * 100:.2f}%")

if __name__ == "__main__":
    run_grid_search(dataset_dir="glaredetection/RGB")