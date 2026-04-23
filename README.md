# 18744 ODD Detection - Clean Code

This clean folder keeps the current mainline code for the ODD detection project.

Current mainline:
- shared DINOv2 backbone
- multi-head prediction for BDD tasks
- road branch for RSCD
- anomaly branch for extreme ODD events
- best road result from staged coarse-to-fine training

---

# 1. Model structure

## Backbone
- DINOv2 ViT-L backbone
- shared visual feature extractor

## BDD branch
The shared feature goes to the BDD adapter and then to these heads:
- `time`
- `scene`
- `visibility`
- `drivable`

These heads are trained in the earlier multitask stage and then kept frozen during later road finetuning.

## Road branch
The shared feature also goes to a separate road adapter and then to these heads:
- `road_condition`
- `road_state`
- `road_severity`

Meaning:
- `road_condition`: 27-class RSCD fine-grained road label
- `road_state`: coarse state label
  - dry
  - wet
  - water
  - snow
  - ice
- `road_severity`: coarse severity label
  - none
  - smooth
  - slight
  - severe

## Anomaly branch
A separate anomaly adapter is added on top of the shared backbone feature.
This branch only serves the anomaly head:

- `anomalies`

Current anomaly classes:
- `none`
- `extreme_weather`
- `road_blockage_hazard`
- `road_structure_failure`

This design keeps the anomaly training isolated from the original BDD heads.
During anomaly training, only:
- `anomaly_adapter`
- `heads["anomalies"]`

are updated.
The backbone, BDD branch, road branch, and drivable head stay frozen.

---

# 2. Best training pipeline

## Stage A: multitask base model
The earlier multitask model was trained first.
This provides:
- trained backbone
- trained BDD heads
- trained drivable head
- initial road branch

Important checkpoint used as the road branch starting point in the original project:
- `checkpoints_multitask_v2_hardboost/best.pt`

This checkpoint is not kept in the current clean folder.
The clean folder keeps only the final mainline checkpoint and the anomaly fine-tuned checkpoint.

## Stage B: road branch finetuning
Several road finetuning directions were tried:
- direct road finetune
- stronger augmentation
- conditioned head
- MixStyle
- semantic soft loss

The best result among these earlier road-only directions came from stronger augmentation on the road branch.

## Stage C: staged coarse-to-fine training
This is the current best road mainline.

### Stage 1
Train only:
- `road_adapter`
- `road_state`
- `road_severity`

Keep frozen:
- backbone
- BDD adapter
- BDD heads
- drivable head

Goal:
- first learn stable coarse road semantics

### Stage 2
Initialize from Stage 1 best checkpoint, then train:
- `road_adapter`
- `road_condition`
- `road_state`
- `road_severity`

Keep frozen:
- backbone
- BDD adapter
- BDD heads
- drivable head

Goal:
- refine fine-grained road condition prediction on top of coarse structure

This staged coarse-to-fine setup gives the best strict road result in the project.

## Stage D: anomaly head training
The anomaly head is initialized from the road mainline checkpoint:
- `checkpoints_road_coarse_to_fine/best.pt`

Then only these parts are trained:
- `anomaly_adapter`
- `heads["anomalies"]`

Light augmentation is used for anomaly training:
- resize to `336 x 336`
- random horizontal flip
- light color jitter

Validation uses resize + normalization only.

---

# 3. Current final checkpoints

## Road mainline checkpoint
- `checkpoints_road_coarse_to_fine/best.pt`

This checkpoint is used for:
- final road evaluation
- original inference pipeline

## Full anomaly-enabled checkpoint
- `checkpoints_anomaly_head/best_model_only.pt`

This checkpoint includes:
- backbone
- BDD heads
- road heads
- anomaly adapter
- anomaly head

This is the current full checkpoint to use for integrated inference.

## Download link
Google Drive link for the integrated checkpoint / shared artifact:
- `https://drive.google.com/file/d/1yWrPM4lI1nOGBmvZ_S_5BIfVXR5Q3bvX/view?usp=sharing`

Note: after the checkpoint saving format fix, future anomaly training runs can directly use `checkpoints_anomaly_head/best.pt` as the full inference checkpoint.

---

# 4. Dataset collection for anomaly head

The anomaly dataset is organized as:

- `none`
- `extreme_weather`
- `road_blockage_hazard`
- `road_structure_failure`

## Source and collection notes
The anomaly dataset was assembled from multiple public sources and manual curation:

- public extreme weather image sources
- selected public videos and frames collected for ODD-like extreme scenes
- manually cleaned and regrouped into the four final categories
- a separate `none` class was collected as non-anomalous negatives

Earlier raw sources included public weather/disaster data such as DAWN and manually sampled images from public videos.
After cleaning and regrouping, the final raw anomaly folder used for training is:

- `anomalies_odd_extreme_raw/`

## Final class counts
- `none`: `1000`
- `extreme_weather`: `1027`
- `road_blockage_hazard`: `787`
- `road_structure_failure`: `468`

Total:
- `3282`

## Split strategy
The anomaly dataset loader uses class-wise splitting, not global random splitting.

With `val_ratio = 0.2`, the current split is:

### train
- `none`: `800`
- `extreme_weather`: `822`
- `road_blockage_hazard`: `630`
- `road_structure_failure`: `375`

Total:
- `2627`

### val
- `none`: `200`
- `extreme_weather`: `205`
- `road_blockage_hazard`: `157`
- `road_structure_failure`: `93`

Total:
- `655`

---

# 5. Final evaluation results

## Original mainline evaluation
Final road mainline evaluation is saved under:

- `eval_outputs/eval_all_heads_coarse_to_fine/summary.md`
- `eval_outputs/eval_all_heads_coarse_to_fine/bdd/bdd_eval.txt`
- `eval_outputs/eval_all_heads_coarse_to_fine/road_aux/road_aux_eval.txt`
- `eval_outputs/eval_all_heads_coarse_to_fine/road_relaxed/road_relaxed_eval.txt`

## Integrated anomaly-enabled evaluation
Integrated evaluation with anomaly head is saved under:

- `eval_outputs/eval_all_heads_with_anomaly_fixed/summary.md`
- `eval_outputs/eval_all_heads_with_anomaly_fixed/anomaly/anomaly_eval.txt`
- `eval_outputs/eval_all_heads_with_anomaly_fixed/bdd/bdd_eval.txt`
- `eval_outputs/eval_all_heads_with_anomaly_fixed/road_aux/road_aux_eval.txt`
- `eval_outputs/eval_all_heads_with_anomaly_fixed/road_relaxed/road_relaxed_eval.txt`

## BDD results

### time
- overall acc: `0.8982`

Per class:
- dawn/dusk: `0.654`
- daytime: `0.876`
- night: `0.978`
- undefined: `0.714`

### scene
- overall acc: `0.7018`

Per class:
- city street: `0.635`
- gas stations: `0.429`
- highway: `0.796`
- parking lot: `0.735`
- residential: `0.856`
- tunnel: `0.852`
- undefined: `0.226`

### visibility
- overall acc: `0.8703`

Per class:
- poor: `0.799`
- medium: `0.706`
- good: `0.923`

### drivable
- mIoU all: `0.6744`
- mIoU fg: `0.5460`
- foreground IoU: `0.4205`

Per class IoU:
- background: `0.9313`
- alternative_drivable: `0.6715`
- direct_drivable: `0.4205`

## Road results

### road_condition
- direct acc: `0.2778`
- infer-style acc: `0.3083`

### road_state
- overall acc: `0.5333`

Per class:
- dry: `0.841`
- wet: `0.170`
- water: `0.932`
- snow: `0.000`
- ice: `0.000`

### road_severity
- overall acc: `0.5722`

Per class:
- none: `0.808`
- smooth: `0.592`
- slight: `0.155`
- severe: `1.000`

### relaxed road metrics
- strict exact: `0.3167`
- relaxed-1: `0.3944`
- relaxed-2: `0.4500`

Relaxed metrics mean:
- relaxed-1 allows `wet <-> water` with same material and severity
- relaxed-2 also allows `slight <-> smooth` with same state and material

## Anomaly results

### integrated eval on anomaly set
- anomalies acc: `0.9951`

Per class:
- none: `0.998` (`998/1000`)
- extreme_weather: `0.996` (`1023/1027`)
- road_blockage_hazard: `0.990` (`779/787`)
- road_structure_failure: `0.996` (`466/468`)

### held-out anomaly val split
This is the stricter anomaly validation result on the class-wise validation split.

- overall acc: `0.9878`

Per class:
- none: `0.9950` (`199/200`)
- extreme_weather: `0.9854` (`202/205`)
- road_blockage_hazard: `0.9809` (`154/157`)
- road_structure_failure: `0.9892` (`92/93`)

---

# 6. Evaluation scripts

## Unified full evaluation
This evaluates:
- BDD heads
- road direct / infer-style
- road relaxed metrics
- anomaly head

```bash
python scripts/eval_all_heads_v2.py \
  --ckpt_path checkpoints_anomaly_head/best_model_only.pt \
  --output_dir eval_outputs/eval_all_heads_with_anomaly_fixed \
  --anomaly_root /home/.../anomalies_odd_extreme_raw
```

## Original road mainline evaluation
```bash
python scripts/eval_all_heads_v2.py \
  --ckpt_path checkpoints_road_coarse_to_fine/best.pt \
  --output_dir eval_outputs/eval_all_heads_coarse_to_fine
```

## Road-only evaluation
```bash
python scripts/eval_road_aux_heads.py \
  --ckpt_path checkpoints_road_coarse_to_fine/best.pt \
  --output_dir eval_outputs/eval_road_aux_heads_coarse_to_fine
```

## Road relaxed evaluation
```bash
python scripts/eval_road_relaxed.py \
  --ckpt_path checkpoints_road_coarse_to_fine/best.pt \
  --output_dir eval_outputs/eval_road_relaxed_coarse_to_fine
```

## Anomaly val-only quick check
A simple anomaly validation check can be run with the anomaly dataset loader and the integrated checkpoint.
This was used to verify the held-out validation accuracy of `0.9878`.

---

# 7. Inference interface

A reusable inference API is provided.

Main file:
- `utils/infer_api.py`

Command-line wrapper:
- `scripts/infer_pipeline_json.py`

This interface is designed for easy downstream use by teammates.

## What the inference API outputs

For each image, the API outputs json fields for:
- `time`
- `scene`
- `visibility`
- `anomalies`
- `road_condition_direct`
- `road_condition_infer`
- `road_state`
- `road_severity`
- `drivable_summary`

`road_condition_infer` uses the current best road inference logic:
- multi-crop evaluation
- auxiliary reranking with `road_state` and `road_severity`
- crop fusion

The anomaly field uses the integrated anomaly-enabled checkpoint and predicts one of:
- `none`
- `extreme_weather`
- `road_blockage_hazard`
- `road_structure_failure`

## Command-line inference example

Single image:
```bash
python scripts/infer_pipeline_json.py \
  --input_path assets/demo.jpg \
  --output_dir infer_outputs/demo_json \
  --ckpt_path checkpoints_anomaly_head/best_model_only.pt
```

Directory inference:
```bash
python scripts/infer_pipeline_json.py \
  --input_path /path/to/images \
  --output_dir infer_outputs/demo_json_dir \
  --ckpt_path checkpoints_anomaly_head/best_model_only.pt \
  --max_images 20
```

Recursive directory inference:
```bash
python scripts/infer_pipeline_json.py \
  --input_path /path/to/images \
  --output_dir infer_outputs/demo_json_dir \
  --ckpt_path checkpoints_anomaly_head/best_model_only.pt \
  --max_images 20 \
  --recursive
```

## Adjustable inference parameters

Road infer parameters:
- `alpha_state`
- `beta_severity`
- `gate_threshold`
- `gate_power`
- `min_mix`

Example:
```bash
python scripts/infer_pipeline_json.py \
  --input_path /path/to/images \
  --output_dir infer_outputs/demo_json_dir \
  --ckpt_path checkpoints_anomaly_head/best_model_only.pt \
  --alpha_state 0.35 \
  --beta_severity 0.10 \
  --gate_threshold 0.60 \
  --gate_power 1.5 \
  --min_mix 0.0
```

## Output structure

```text
infer_outputs/demo_json_dir/
  per_image/
    xxx.json
    yyy.json
  results_all.json
  run_info.json
```

- `per_image/*.json`: one json per image
- `results_all.json`: all results together
- `run_info.json`: run-level metadata

## checkpoint
https://drive.google.com/file/d/1yWrPM4lI1nOGBmvZ_S_5BIfVXR5Q3bvX/view?usp=sharing


## Python API example

Single image:
```python
from utils.infer_api import load_pipeline, infer_single_image_path

pipe = load_pipeline(
    ckpt_path="checkpoints_anomaly_head/best_model_only.pt"
)

result = infer_single_image_path(
    image_path="/path/to/demo.jpg",
    pipeline=pipe,
)

print(result["prediction"]["time"]["label"])
print(result["prediction"]["anomalies"]["label"])
print(result["prediction"]["road_condition_infer"]["label"])
```

Directory:
```python
from utils.infer_api import load_pipeline, infer_path, save_results_to_dir

pipe = load_pipeline(
    ckpt_path="checkpoints_anomaly_head/best_model_only.pt"
)

results = infer_path(
    input_path="/path/to/images",
    pipeline=pipe,
    max_images=20,
)

save_results_to_dir(
    results=results,
    output_dir="infer_outputs/api_demo",
    pipeline=pipe,
    input_path="/path/to/images",
)
```

# 8. Stage 2 documentation starts below

# Stage-2 Tiebreaker: Multi-Modal Post-Fusion for ODD Detection
 
## Overview
 
Stage-2 is a post-processing pipeline that corrects stage-1 vision model outputs using multi-modal feature fusion. It consists of two serial components:
 
1. **MLP Tiebreaker** — a lightweight MLP (60→128→64→6 heads, ~20k params) that fuses stage-1 predictions with CLIP-inferred virtual sensor data (temperature, humidity, clock time) to correct weather, time-of-day, scene, and anomaly classifications.
2. **Soft-Mask Guardrail** — a rule-based engine (no learned parameters) that uses MLP-corrected weather probabilities to reweight the 27-class road surface prediction and 3-class visibility prediction via plausibility masking.
### Key Results (Validation Set, n=2965)
 
| Head | Metric | Stage-1 | Stage-2 | Delta |
|------|--------|---------|---------|-------|
| Fog | F1 | 0.80 | 0.99 | +0.19 |
| Rain | F1 | 0.30 | 0.81 | +0.51 |
| Snow | F1 | 0.16 | 0.75 | +0.59 |
| Time | Accuracy | 89.1% | 95.7% | +6.6% |
| Scene | Accuracy | 71.6% | 83.3% | +11.7% |
 
Results validated on ACDC-only subset (unambiguous labels): fog F1=0.995, rain F1=0.979, snow F1=0.988.
 
---
 
## Architecture
 
```
Stage-1 Models (6 models, unchanged)
├── [0] CLIP + GeoCLIP virtual sensors (temperature, humidity, clock_time, location)
├── [1] SegFormer cloud detection (cloud_fraction)
├── [2] SegFormer glare detection (glare_ratio)
├── [3] ResNet18 weather (fog/rain/snow severity)
├── [4] DINOv2 ODD multi-head (time, scene, visibility, anomalies, road_condition, road_state)
└── [5] YOLO11m (car/pedestrian/bicycle density, work_zone)
         │
         ▼
    join_jsons.py → merged_json/*.json
         │
         ▼
Stage-2 Post-Processing (new)
├── MLP Tiebreaker
│   ├── Input: 60-dim feature vector (53 effective, 7 deprecated)
│   ├── Output: corrected fog/rain/snow (sigmoid), time/scene/anomalies (softmax)
│   └── Checkpoint: models/tiebreaker/tiebreaker_best.pt
└── Soft-Mask Guardrail
    ├── Input: MLP weather probs + stage-1 road_condition(27) + visibility(3) softmax
    ├── Method: plausibility vector × softmax → renormalize
    └── Output: corrected road_condition(27), visibility(3), aggregated road_state(5)
         │
         ▼
    merged_json/*.json  →  "stage2" field appended
```
 
### MLP Input Layout (60 dims)
 
| Index | Content | Source |
|-------|---------|--------|
| [0] | cloud_fraction | SegFormer |
| [1] | fog_severity | ResNet18 |
| [2] | rain_severity | ResNet18 |
| [3] | snow_severity | ResNet18 |
| [4] | glare_ratio | SegFormer |
| [5:9] | time softmax (4) | DINOv2 |
| [9:16] | scene softmax (7) | DINOv2 |
| [16:19] | visibility softmax (3) | DINOv2 |
| [19:23] | anomalies softmax (4) | DINOv2 |
| [23:50] | road_condition softmax (27) | DINOv2 |
| [50] | temperature normalized | CLIP |
| [51] | humidity normalized | CLIP |
| [52] | clock_time normalized | CLIP |
| [53:59] | OSM one-hot (6) | DEPRECATED — hardcoded 0 |
| [59] | geoclip_confidence | DEPRECATED — hardcoded 0 |
 
### Guardrail Parameters
 
```
α = 0.7   (rain suppression on dry state)
β = 0.8   (rain boost on wet state, baseline 0.3)
γ = 0.7   (rain boost on water state, no baseline)
δ = 0.5   (snow-to-ice coupling)
FLOOR = 0.2  (minimum plausibility for dry/wet/snow states only; water/ice have no floor)
```
 
Plausibility is normalized by class count per state to prevent states with more sub-classes (e.g., dry with 8) from dominating aggregated probability mass.
 
---
 
## File Structure
 
```
models/tiebreaker/
├── tiebreaker_mlp.py        # MLP model definition (TiebreakerMLP)
├── tiebreakers_guard.py     # Guardrail soft-mask logic
├── tiebreaker_best.pt       # Trained MLP checkpoint
├── eval_guardrail.py        # Guardrail evaluation (consistency + HTML report)
└── eval_mlp.py              # MLP stage-1 vs stage-2 comparison
 
scripts/
├── data_harvester.py        # Assembles training data from 3 datasets → .pt
├── run_models.py            # Full pipeline (stage-1 + merge + stage-2)
├── infer_single.py          # Single-image inference (stage-1 + stage-2)
└── join_jsons.py            # Legacy standalone merge script (now inlined in run_models.py)
 
data/
└── tiebreaker_train.pt      # Training data {X: [19772, 60], Y: [19772, 6]}

─ train_mlp.py             # MLP training script
```
 
---
 
## Usage
 
### Full Pipeline (Stage-1 + Stage-2)
 
```bash
# Place input images in source_images/
python scripts/run_models.py
# Output: outputs/merged_json/*.json (with "stage2" field)
```
 
### Single Image Inference
 
```bash
# Full pipeline on one image
python scripts/infer_single.py path/to/image.jpg
 
# Stage-2 only (skip stage-1, use existing merged JSON)
# NOTE: You may need to modify the paths to the merged JSON to get this script working!
python scripts/infer_single.py path/to/image.jpg --skip_stage1
```
 
### Evaluation
 
```bash
# MLP stage-1 vs stage-2 comparison
python models/tiebreaker/eval_mlp.py
 
# Guardrail evaluation (consistency checks + HTML report)
python -m models.tiebreaker.eval_guardrail
 
# Guardrail sanity check (synthetic scenarios)
python -m models.tiebreaker.tiebreakers_guard
```
 
### Training
 
```bash
# Step 1: Harvest training data from 3 datasets
python scripts/data_harvester.py --skip_osm
 
# Step 2: Train MLP
python scripts/train_mlp.py
 
# Step 3: Copy checkpoint
cp checkpoints_tiebreaker/tiebreaker_best.pt models/tiebreaker/tiebreaker_best.pt
```
 
---
 
## Training Data
 
Three datasets unified via per-head label masking (Y=-1 for missing labels):
 
| Dataset | Samples | fog | rain | snow | time | scene | anomalies |
|---------|---------|-----|------|------|------|-------|-----------|
| BDD100K val | 10,000 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| ACDC val | 2,356 | ✓ | ✓ | ✓ | partial | ✗ | ✗ |
| ROADwork | 7,416 | ✗ | ✓ | ✓ | ✓ | partial | ✗ |
 
Split: 85/15 train/val, seed=42. Val set: 2,965 samples.
 
---
 
## Known Limitations
 
1. **CLIP virtual sensor noise** — CLIP-inferred temperature can be significantly inaccurate (e.g., estimating 6°C on a warm autumn day), causing false snow predictions. In production, real sensor inputs would eliminate this.
2. **Guardrail cannot override saturated softmax** — Soft-mask reweighting uses multiplication; when stage-1 assigns >95% confidence to a single class, the plausibility ratio (~3-5x) is insufficient to flip the argmax. Potential future fix: temperature scaling before reweighting.
3. **No direct GT for road_condition and visibility** — Evaluation relies on indirect proxies (weather-surface consistency checks, road_state cross-reference, manual inspection). Quantitative accuracy metrics are not available.
4. **RSCD domain gap** — The 27-class road surface classifier was trained on close-up road images, not dashcam perspectives. Guardrail can correct the state-level prediction (dry→wet) but sub-class texture predictions (asphalt vs concrete) remain unreliable.
5. **Label policy inconsistency** — BDD100K labels post-rain scenes as "rainy" while ACDC only labels active precipitation. Validated on ACDC-only subset to confirm improvements are not artifacts of label noise.
6. **Deprecated features** — OSM 6-dim one-hot (Overpass API unreachable) and GeoCLIP confidence (meaningless without OSM) are hardcoded to zero. 7 of 60 input dimensions carry no signal.
---
 
## Output Format
 
Stage-2 results are appended to each merged JSON under the `"stage2"` key. All existing fields are preserved unchanged.
 
```json
{
    "cloud_detection": { "..." },
    "glare": { "..." },
    "synth_outputs": { "..." },
    "weather": { "..." },
    "yuheng": { "..." },
    "yolo": { "..." },
    "stage2": {
        "weather_corrected": {
            "fog": 0.001,
            "rain": 0.951,
            "snow": 0.003
        },
        "time_corrected": {
            "label": "daytime",
            "class_id": 1
        },
        "scene_corrected": {
            "label": "city street",
            "class_id": 0
        },
        "anomalies_corrected": {
            "label": "none",
            "class_id": 0
        },
        "road_condition_corrected": {
            "label": "wet_asphalt_slight",
            "class_id": 19,
            "confidence": 0.547,
            "probabilities": ["...27 floats..."]
        },
        "road_condition_aggregated": {
            "dry": 0.221,
            "wet": 0.604,
            "water": 0.148,
            "snow": 0.028,
            "ice": 0.000
        },
        "visibility_corrected": {
            "label": "poor",
            "class_id": 0,
            "confidence": 0.730,
            "probabilities": [0.730, 0.048, 0.222]
        }
    }
}
```
## Model Weights

then, make sure the following weights are available:
  models/glare/custom_glare_model/config.json
  models/glare/custom_glare_model/model.safetensors
  models/weather/weather_resnet18_best.pth
  models/yolo/yolo_traffic_workzone.pt
  models/yolo/density_thresholds.json
  models/yuheng/odd_full_infer_best.pt

your input images should be placed at source_images/
acceptable formats are .jpg .jpeg .png

## Visualization
outputs are stored in stage1_outputs/,
one json per category per image, arranged in 6 separate directories
  cloud_detection/
  glare/
  synth_outputs/
  weather/
  yolo/
  yuheng/
corresponding to the 6 stages in run_models.py


## Merging results
Run scripts/join_jsons.py to merge the separate jsons into one single json per image,
merged jsons are stored in stage1_outputs/merged_json/


Download YOLO weights, density threshold file and Tiebreaker weights from 
https://drive.google.com/drive/folders/11kQnLEV514uu7eh7cqKJ21t7VdRhVbFI?usp=sharing

and place YOLO and density threshold files in models/yolo/,

place Tiebreaker weights in models/tiebreaker.


---
