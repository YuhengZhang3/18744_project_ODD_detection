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

The merged **Stage1** inference script is located at scripts/run_models.py,

## Preparation
to run it you need to download the CLIP weights from the project root directory
wget -c https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors -P ./cache/clip-vit-large-patch14/

then, make sure the following weights are available:
  models/custom_glare_model/config.json
  models/custom_glare_model/model.safetensors
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



---
