import os
import cv2
import numpy as np
import glob
import json

def analyze_precipitation(image_bgr, sky_mask=None):
    """
    Uses Frequency (FFT) and Spatial Noise analysis to determine 
    precipitation type and severity.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # ---------------------------------------------------------
    # 1. SEVERITY: Spatial High-Frequency Noise Isolation
    # ---------------------------------------------------------
    background_removed = cv2.medianBlur(gray, 5)
    high_freq_noise = cv2.absdiff(gray, background_removed)
    
    if sky_mask is not None:
        valid_noise = high_freq_noise[sky_mask == 255]
    else:
        # Fallback: analyze the upper third of the image
        valid_noise = high_freq_noise[:h//3, :]
        
    if valid_noise.size == 0:
        return {"precipitation": "none", "severity": 0.0}
        
    noise_level = np.mean(valid_noise)
    severity_score = max(0.0, min((noise_level - 1.0) / 7.0, 1.0))
    
    if severity_score < 0.05:
        return {"precipitation": "none", "severity": 0.0}

    # ---------------------------------------------------------
    # 2. TYPE: Fast Fourier Transform (FFT) Directional Analysis
    # ---------------------------------------------------------
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    cy, cx = h // 2, w // 2
    cv2.circle(magnitude_spectrum, (cx, cy), 30, 0, thickness=-1)
    
    high_freq_variance = np.var(magnitude_spectrum[magnitude_spectrum > 0])
    
    if high_freq_variance > 1500: 
        precip_type = "rain"
    else:
        precip_type = "snow"
        
    return {
        "precipitation": precip_type,
        "severity": round(float(severity_score), 3),
        "debug_metrics": {
            "spatial_noise_mean": round(float(noise_level), 3),
            "fft_variance": round(float(high_freq_variance), 3)
        }
    }

def process_images(input_dir="../../source_images", json_dir="output_json"):
    # 1. Setup directory
    os.makedirs(json_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, "*.[jp][pn]*[g]")) 

    if not image_paths:
        print(f"No images found in {input_dir}.")
        return

    print(f"Processing {len(image_paths)} images for precipitation...")

    # 2. Process each image
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            continue
            
        # 3. Analyze the weather
        results = analyze_precipitation(img_bgr)
        
        # 4. Save to JSON
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(json_dir, json_filename)
        
        # Merge with existing JSON data if it exists (e.g., from sky_segmentation)
        existing_data = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        existing_data.update(results)
        with open(json_path, 'w') as f:
            json.dump(existing_data, f, indent=4)

        print(f"Updated JSON: {json_path}")

if __name__ == "__main__":
    process_images(input_dir="../../source_images", json_dir="output_json")