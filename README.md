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