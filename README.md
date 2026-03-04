# 18744 Project - ODD Detection
Code base for our 18744 ODD detection project.

## Model
Backbone  
- DINOv2 ViT-L/14 from dinov2  
- Input: 336 x 336 RGB image  
- Output: CLS token and patch tokens

Adapter  
- Shared MLP on top of DINOv2 features  
- LayerNorm + Linear + GELU  
- Output dim: 1024

Heads (multi-head classifier)  

Global heads(use CLS feature)  
- weather  
- lighting 
- time  
- traffic  
- road_condition
- scene   

Attentive heads (patch features with small attention pooling)  
- anomalies 

Each head is independent. Backbone and adapter are shared.  
New heads can be added later without changing existing heads.


## Current progress: BDD100K time + scene baseline

We first focus on two ODD dimensions that can be directly trained from BDD100K:
- time of day
- scene type

### Labels

We reuse the original BDD100K attributes:

- **Time of day (`time` head)**  
  - dawn/dusk  
  - daytime  
  - night  
  - undefined  

- **Scene type (`scene` head)**  
  - city street  
  - gas stations  
  - highway  
  - parking lot  
  - residential  
  - tunnel  
  - undefined  

These labels are parsed from the JSON files under `100k_label/{train,val}`.

### Model

We use a shared ViT backbone and multiple classifier heads:

- backbone: DINOv2 ViT-L/14 (frozen)
- one time head: 3 + 1 classes (dawn/dusk, daytime, night, undefined)
- one scene head: 6 + 1 classes (city street, gas stations, highway, parking lot, residential, tunnel, undefined)
- heads are simple MLPs on top of the global DINOv2 feature

Code:
- `models/odd_model.py` – ODDModel with configurable heads
- `configs/odd_config.py` – defines time/scene heads used in this stage
- `data/bdd_dataset.py` – BDDDTimeScene dataset and collate
- `scripts/train_time_scene_joint.py` – joint training script (epoch-based, ckpt + resume)
- `scripts/eval_time_scene.py` – evaluation on BDD100K val set

### Training setup

- dataset: BDD100K `train` / `val` (100k split)
- optimizer: AdamW, lr = 3e-4, weight decay = 0.05
- scheduler: cosine LR, 15 epochs
- batch size: 64 on a single A100 80GB
- backbone frozen, only heads are trained

### Results on BDD100K val

**Time of day (overall ~93–94% acc)**

- dawn/dusk: 0.546  (425 / 778)  
- daytime:   0.956  (5029 / 5258)  
- night:     0.985  (3871 / 3929)  
- undefined: 0.714  (25 / 35)  

Time head is very strong on daytime and night. Dawn/dusk is harder and has fewer samples, which is expected.

**Scene type (overall ~80% acc)**

- city street: 0.913  (5582 / 6112)  
- gas stations: 0.571 (4 / 7)  
- highway: 0.693      (1731 / 2499)  
- parking lot: 0.571  (28 / 49)  
- residential: 0.494  (619 / 1253)  
- tunnel: 0.815       (22 / 27)  
- undefined: 0.019    (1 / 53)  

City street is well recognized. Highway is moderate. Residential is more ambiguous and often confused with city street. The undefined class is noisy and mostly treated as a “don’t-care” bucket.

This stage gives us a solid baseline for time and scene heads. Next steps will add more ODD dimensions (e.g. visibility, road condition, weather) on top of the same DINOv2 backbone.

## Visibility labeling on BDD100K

Goal: build a 3-level visibility label (`vis ∈ {0,1,2}` = poor / medium / good) as one ODD dimension.

### Source labels

Use BDD100K attributes:

- `weather` (clear, rainy, foggy, snowy, overcast, …)
- `timeofday` (dawn/dusk, daytime, night, undefined)

For analysis, images are grouped as: clear_day, clear_night, rainy, foggy, snowy.

### CV features

For each image, simple, dataset-agnostic features:

- global grayscale contrast, global edge density
- near-road ROI (lower-middle): contrast
- far-road ROI (mid-height): contrast
- sky/top region: contrast
- basic blur / haze scores (Laplacian variance, dark-channel style score)

No network training is used here; only low-level image statistics.

### Current rule

Visibility label:

- `vis = 2` (good)  
  - mainly daytime / dawn / dusk  
  - near-road and far-road contrast above thresholds

- `vis = 1` (medium)  
  - mainly night scenes  
  - near-road contrast high, far-road and sky contrast low

- `vis = 0` (poor)  
  - foggy scenes, or  
  - both near-road and far-road contrast very low

Implementation:

- `scripts/label_visibility_bdd.py`  
  - read BDD100K images + JSON  
  - write extra JSON files to `datasets/visibility_labels/{train,val,test}` with:
    - `visibility`, `weather`, `timeofday`, `near_contrast`, `far_contrast`

- `scripts/vis_visibility_from_labels.py`  
  - sample images per visibility level from val split  
  - save grid `vis_visibility_samples_val.png` for manual sanity check

Rough manual check on samples shows about 80–85% agreement with human judgment.  
Thresholds are currently tuned on BDD100K and can be re-calibrated (e.g., via quantiles or a small classifier) for other datasets.

### Visibility head

- Use BDD100K image-level labels (`weather`, `timeofday`) and simple CV rules
  (near/far contrast in gray image) to build a 3-class visibility label:
  - 0 = poor, 1 = medium, 2 = good.
- Labels are stored as JSON under `visibility_labels/{train,val,test}/*_vis.json`.
- Train a separate head `visibility` on top of frozen DINOv2 features
  using `scripts/train_visibility_head.py`.

Current validation results on BDD100K val:

- overall acc: **87.9%**
- per class:
  - poor: 80.4% (2245 samples)
  - medium: 68.2% (1153 samples)
  - good: 94.0% (6602 samples)