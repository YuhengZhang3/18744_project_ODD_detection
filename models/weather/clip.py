import os
import json
import glob
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def predict_weather(input_dir="../../datasets/acdc", json_dir="clip_json"):
    os.makedirs(json_dir, exist_ok=True)
    
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LARGE CLIP model on {device.type.upper()}...")

    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    # The 7-point severity scale
    weights = torch.tensor([0.0, 0.15, 0.33, 0.50, 0.66, 0.85, 1.0], device=device)

    # 1. DEFINE PROMPT TEMPLATES FOR ENSEMBLING
    templates = [
        "A dashcam photo of {}",
        "A view from a car driving in {}",
        "A street level view showing {}",
        "A photo of {}",
        "A driving scene with {}"
    ]

    # 2. DEFINE DISENTANGLED BASE DESCRIPTIONS
    weather_descriptions = {
        "fog": [
            "a road with crisp, perfect visibility and absolutely no fog",
            "a road with a very faint, distant haze",
            "a road with light fog reducing distant visibility",
            "a road in moderate fog with a gray atmospheric haze",
            "a road in heavy fog, thick white haze obscuring the background",
            "a road in very dense fog, highly restricted visibility",
            "a road in severe, blinding zero-visibility fog whiteout"
        ],
        "rain": [
            "a road where it is absolutely not raining, no raindrops at all",
            "a road with trace drizzle, a few scattered raindrops",
            "a road in light rain, slightly wet asphalt",
            "a road in moderate rain, wet reflective roads and falling raindrops",
            "a road in heavy rain, lots of splashing water and raindrops",
            "a road in very heavy rain, covered in water and heavy splashing",
            "a road in a severe, torrential rain storm, flooded with water"
        ],
        "snow": [
            "a road with absolutely no snow on the ground or in the air",
            "a road with trace snow flurries but clear asphalt",
            "a road in light snow, small amounts of white snow accumulating",
            "a road in moderate snow, roads and surroundings covered in white snow",
            "a road in heavy snow, thick white snow everywhere",
            "a road covered in very heavy deep snow, active snowfall",
            "a road in severe blizzard whiteout conditions, completely buried in snow"
        ]
    }

    print("Pre-computing ensembled text embeddings...")
    ensembled_text_features = {}

    # 3. PRE-COMPUTE ENSEMBLES
    with torch.no_grad():
        for weather_type, descriptions in weather_descriptions.items():
            bucket_embeddings = []
            
            for desc in descriptions:
                # Combine the 5 templates with the 1 description
                sentences = [template.format(desc) for template in templates]
                
                # Tokenize and encode
                text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
                text_outputs = model.get_text_features(**text_inputs)
                
                # --- FIX: Handle Hugging Face version differences ---
                if isinstance(text_outputs, torch.Tensor):
                    text_features = text_outputs
                else:
                    text_features = text_outputs.pooler_output
                
                # Normalize, average the 5 templates together, and normalize the result again
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                mean_features = text_features.mean(dim=0)
                mean_features = mean_features / mean_features.norm(p=2, dim=-1, keepdim=True)
                
                bucket_embeddings.append(mean_features)
            
            # Stack the 7 severity buckets into a single tensor for this weather type
            ensembled_text_features[weather_type] = torch.stack(bucket_embeddings)

    print(f"Running highly optimized inference on {len(image_paths)} images...")

    # 4. PROCESS IMAGES
    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            try:
                img_pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
            
            # Extract image features
            image_inputs = processor(images=img_pil, return_tensors="pt").to(device)
            image_outputs = model.get_image_features(**image_inputs)
            
            # --- FIX: Handle Hugging Face version differences ---
            if isinstance(image_outputs, torch.Tensor):
                image_features = image_outputs
            else:
                image_features = image_outputs.pooler_output
                
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # CLIP's internal scaling factor
            logit_scale = model.logit_scale.exp()
            
            weather_metrics = {}

            # Compare the single image embedding against our pre-computed text ensembles
            for weather_type in ["fog", "rain", "snow"]:
                
                text_embeds = ensembled_text_features[weather_type]
                
                # Calculate Cosine Similarity (Image @ Text.T)
                logits = logit_scale * image_features @ text_embeds.T
                
                # Convert logits to probabilities
                probs = logits.softmax(dim=-1).squeeze()
                
                # Calculate weighted score (0.0 to 1.0)
                severity_score = torch.sum(probs * weights).item()
                weather_metrics[f"{weather_type}_severity"] = round(severity_score, 4)

            # Save to JSON
            json_filename = f"{basename}.json"
            json_path = os.path.join(json_dir, json_filename)
            
            output_data = {"weather": weather_metrics}
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"Processed: {json_filename} | F: {weather_metrics['fog_severity']:.3f} | R: {weather_metrics['rain_severity']:.3f} | S: {weather_metrics['snow_severity']:.3f}")

if __name__ == "__main__":
    predict_weather()