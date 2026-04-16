import os
import json
import glob
import cv2
import numpy as np

# Adjust project_root if you are running this from a different location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JSON_ROOT = os.path.join(project_root, "outputs")
MERGED_OUT_DIR = os.path.join(JSON_ROOT, "merged_json")
GLARE_DIR = os.path.join(JSON_ROOT, "glare")

def main():
    if not os.path.exists(JSON_ROOT):
        print(f"Error: Could not find outputs directory at {JSON_ROOT}")
        return

    # Ensure the target output directory exists
    os.makedirs(MERGED_OUT_DIR, exist_ok=True)

    # 1. Identify all subdirectories to merge
    # Excluding 'glare' (handled manually) and 'merged_json' (to avoid recursive reading)
    subdirs = [
        d for d in os.listdir(JSON_ROOT) 
        if os.path.isdir(os.path.join(JSON_ROOT, d)) 
        and d not in ["merged_json", "glare"]
    ]
    
    print(f"Directories to merge (JSON): {subdirs}")
    print(f"Processing images from: ['glare']")

    # 2. Collect all unique basenames across all valid subdirectories and the glare folder
    basenames = set()
    
    # Collect from JSON folders
    for subdir in subdirs:
        json_files = glob.glob(os.path.join(JSON_ROOT, subdir, "*.json"))
        for jf in json_files:
            basename = os.path.splitext(os.path.basename(jf))[0]
            basenames.add(basename)
            
    # Collect from Glare image folder
    if os.path.exists(GLARE_DIR):
        glare_files = glob.glob(os.path.join(GLARE_DIR, "*.png"))  # Assuming .png for masks
        for gf in glare_files:
            basename = os.path.splitext(os.path.basename(gf))[0]
            basenames.add(basename)
    
    basenames = sorted(list(basenames))
    print(f"Found {len(basenames)} unique frames to process.")

    # 3. Build and save the merged dictionary per frame
    for basename in basenames:
        frame_data = {}
        
        # --- Handle JSON subdirectories ---
        for subdir in subdirs:
            json_path = os.path.join(JSON_ROOT, subdir, f"{basename}.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        frame_data[subdir] = json.load(f)
                except json.JSONDecodeError:
                    frame_data[subdir] = {"error": "Invalid JSON format"}
            else:
                frame_data[subdir] = None 
                
        # --- Handle Glare image ---
        glare_path = os.path.join(GLARE_DIR, f"{basename}.png")
        if os.path.exists(glare_path):
            # Read image in grayscale
            mask = cv2.imread(glare_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # Calculate percentage of glare (pixels >= 128 considered white/glare)
                glare_pixels = np.sum(mask >= 128)
                total_pixels = mask.size
                glare_percentage = float(glare_pixels / total_pixels)
                
                frame_data["glare"] = {
                    "glare_percentage": round(glare_percentage, 6)
                }
            else:
                frame_data["glare"] = {"error": "Failed to read glare image file"}
        else:
            frame_data["glare"] = None
                
        # 4. Save to its own JSON file in the merged_json folder
        output_file = os.path.join(MERGED_OUT_DIR, f"{basename}.json")
        with open(output_file, 'w') as f:
            json.dump(frame_data, f, indent=4)
    
    print(f"Successfully generated {len(basenames)} merged files in: {MERGED_OUT_DIR}")

if __name__ == "__main__":
    main()