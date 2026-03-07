import os
import cv2
import glob

def view_results_gui(orig_dir="../../source_images", pred_dir="glare_predictions"):
    # Find all the prediction images and sort them alphabetically
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "pred_*.*")))
                 
    if not pred_paths:
        print(f"No predictions found in '{pred_dir}/'. Run your test script first!")
        return
        
    print(f"Found {len(pred_paths)} images. Launching GUI...")
    print("CONTROLS:")
    print("  'n' -> Next image")
    print("  'b' -> Previous image")
    print("  'q' -> Quit")

    idx = 0
    total_images = len(pred_paths)
    
    # Target resolution for EACH image (Width, Height)
    target_size = (800, 450)

    while True:
        pred_path = pred_paths[idx]
        filename = os.path.basename(pred_path).replace("pred_", "")
        orig_path = os.path.join(orig_dir, filename)
        
        if not os.path.exists(orig_path):
            print(f"Warning: Could not find original image for {filename}")
            idx = (idx + 1) % total_images
            continue
            
        # Load images
        orig_img = cv2.imread(orig_path)
        pred_img = cv2.imread(pred_path)
        
        # --- Force both images to 800x450 ---
        orig_img = cv2.resize(orig_img, target_size)
        pred_img = cv2.resize(pred_img, target_size)
            
        # Add a black background rectangle for readable text
        cv2.rectangle(orig_img, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.rectangle(pred_img, (0, 0), (320, 40), (0, 0, 0), -1)

        # Add text labels (scaled down slightly for the 800px width)
        cv2.putText(orig_img, f"Original ({idx+1}/{total_images})", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(pred_img, "SegFormer Prediction", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Stitch them horizontally -> Final window will be 1600x450
        combined = cv2.hconcat([orig_img, pred_img])

        # Show the image
        cv2.imshow("Glare Detection Viewer", combined)
        
        # Wait for keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):    # Quit
            break
        elif key == ord('n'):  # Next
            idx = (idx + 1) % total_images
        elif key == ord('b'):  # Back
            idx = (idx - 1) % total_images

    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_results_gui()