import os
import json
import glob

# Adjust project_root if you are running this from a different location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JSON_ROOT = os.path.join(project_root, "outputs")
MERGED_OUT_DIR = os.path.join(JSON_ROOT, "merged_json")

def main():
    if not os.path.exists(JSON_ROOT):
        print(f"Error: Could not find outputs directory at {JSON_ROOT}")
        return

    # Ensure the target output directory exists
    os.makedirs(MERGED_OUT_DIR, exist_ok=True)

    # 1. Identify all subdirectories to merge
    # Excluding 'glare' (as requested) and 'merged_json' (to avoid recursive reading)
    subdirs = [
        d for d in os.listdir(JSON_ROOT) 
        if os.path.isdir(os.path.join(JSON_ROOT, d)) 
        and d not in ["glare", "merged_json"]
    ]
    
    print(f"Directories to merge: {subdirs}")

    # 2. Collect all unique JSON basenames across all valid subdirectories
    basenames = set()
    for subdir in subdirs:
        json_files = glob.glob(os.path.join(JSON_ROOT, subdir, "*.json"))
        for jf in json_files:
            basename = os.path.splitext(os.path.basename(jf))[0]
            basenames.add(basename)
    
    basenames = sorted(list(basenames))
    print(f"Found {len(basenames)} unique frames to process.")

    # 3. Build and save the merged dictionary per frame
    for basename in basenames:
        frame_data = {}
        for subdir in subdirs:
            json_path = os.path.join(JSON_ROOT, subdir, f"{basename}.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        frame_data[subdir] = json.load(f)
                except json.JSONDecodeError:
                    frame_data[subdir] = {"error": "Invalid JSON format"}
            else:
                # If a frame is missing from a specific model's output
                frame_data[subdir] = None 
                
        # 4. Save to its own JSON file in the merged_json folder
        output_file = os.path.join(MERGED_OUT_DIR, f"{basename}.json")
        with open(output_file, 'w') as f:
            json.dump(frame_data, f, indent=4)
    
    print(f"Successfully generated {len(basenames)} merged files in: {MERGED_OUT_DIR}")

if __name__ == "__main__":
    main()