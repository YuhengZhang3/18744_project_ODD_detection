import os
import cv2
import glob

def view_clarity_gui(res_dir="clarity_results"):
    # Find all the clarity result images
    res_paths = sorted(glob.glob(os.path.join(res_dir, "*")))
                 
    if not res_paths:
        print(f"No results found in '{res_dir}/'. Run test_clarity.py first!")
        return
        
    print(f"Found {len(res_paths)} images. Launching GUI...")
    print("CONTROLS: 'n' = Next | 'b' = Back | 'q' = Quit")

    idx = 0
    total_images = len(res_paths)

    while True:
        res_path = res_paths[idx]
        res_img = cv2.imread(res_path)
        
        # Resize to a comfortable viewing size (e.g., 1024x576)
        res_img = cv2.resize(res_img, (1024, 576))
        h, w = res_img.shape[:2]
            
        # Draw background box for the Image Counter in the TOP RIGHT
        counter_text = f"{idx+1}/{total_images}"
        cv2.rectangle(res_img, (w - 120, 0), (w, 35), (0, 0, 0), -1)
        cv2.putText(res_img, counter_text, (w - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the single image
        cv2.imshow("Scene Clarity Viewer", res_img)
        
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
    view_clarity_gui()