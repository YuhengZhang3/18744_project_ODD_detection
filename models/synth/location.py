import os
import json
import glob
import torch
import types
import reverse_geocoder as rg
from geoclip import GeoCLIP

def append_geoclip_location(input_dir="../../source_images", json_dir="output_json"):
    os.makedirs(json_dir, exist_ok=True)
    
    # 1. Collect all images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    # 2. Setup Device and GeoCLIP Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GeoCLIP model on {device.type.upper()}...")
    
    model = GeoCLIP().to(device)
    model.eval()

    # --- MONKEY PATCH FIX FOR HUGGING FACE CONFLICT ---
    # Intercepts the linear layer in the image encoder to safely unpack the tensor
    for module in model.image_encoder.modules():
        if isinstance(module, torch.nn.Linear):
            original_forward = module.forward
            
            def patched_forward(self, x, orig=original_forward):
                # If the input is the Hugging Face output object, extract the raw tensor
                if hasattr(x, "pooler_output") and x.pooler_output is not None:
                    x = x.pooler_output
                elif hasattr(x, "last_hidden_state"):
                    x = x.last_hidden_state[:, 0, :]
                return orig(x)
                
            module.forward = types.MethodType(patched_forward, module)
    # -------------------------------------------------

    print(f"Running GeoCLIP inference for {len(image_paths)} images...")

    # 3. Process Images
    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            json_path = os.path.join(json_dir, f"{basename}.json")
            
            try:
                # Get exact coordinates
                top_pred_gps, top_pred_prob = model.predict(img_path, top_k=1)
                best_lat = float(top_pred_gps[0][0])
                best_lon = float(top_pred_gps[0][1])
                best_prob = float(top_pred_prob[0])
                
                # Reverse Geocode to nearest city (Offline lookup)
                rg_result = rg.search((best_lat, best_lon), verbose=False)[0]
                city = rg_result.get('name', 'Unknown City')
                state = rg_result.get('admin1', 'Unknown State')
                country = rg_result.get('cc', 'Unknown Country')
                
                readable_location = f"{city}, {state}, {country}"

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            # Load existing JSON
            output_data = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        output_data = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            if "sensors" not in output_data:
                output_data["sensors"] = {}

            # Append the GPS location AND the nearest city
            output_data["sensors"]["location"] = {
                "lat": round(best_lat, 6),
                "lon": round(best_lon, 6),
                "nearest_city": readable_location,
                "geoclip_confidence": round(best_prob, 4)
            }
            
            # Save the updated payload
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"Processed: {filename} | Loc: {readable_location} | Conf: {best_prob:.4f}")

if __name__ == "__main__":
    append_geoclip_location()