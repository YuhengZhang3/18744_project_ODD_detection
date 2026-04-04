# 18744 ODD Detection - Clean Code

This clean folder keeps the current mainline code for the ODD detection project.

Current mainline:
- shared DINOv2 backbone
- multi-head prediction for BDD tasks
- road branch for RSCD
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
The clean folder keeps only the final mainline checkpoint.

## Stage B: road branch finetuning
Several road finetuning directions were tried:
- direct road finetune
- stronger augmentation
- conditioned head
- MixStyle
- semantic soft loss

The best result among these earlier road-only directions came from stronger augmentation on the road branch.

## Stage C: staged coarse-to-fine training
This is the current best mainline.

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

---

# 3. Current final checkpoint

The clean folder keeps only the final best checkpoint:

- `checkpoints_road_coarse_to_fine/best.pt`

This checkpoint is used for:
- final evaluation
- inference API
- downstream integration

---

# 4. Final evaluation results

Final evaluation is saved under:

- `eval_outputs/eval_all_heads_coarse_to_fine/summary.md`
- `eval_outputs/eval_all_heads_coarse_to_fine/bdd/bdd_eval.txt`
- `eval_outputs/eval_all_heads_coarse_to_fine/road_aux/road_aux_eval.txt`
- `eval_outputs/eval_all_heads_coarse_to_fine/road_relaxed/road_relaxed.txt`

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

---

# 5. Evaluation scripts

## Unified full evaluation
This evaluates:
- BDD heads
- road direct / infer-style
- road relaxed metrics

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

## Coarse-only road evaluation
```bash
python scripts/eval_road_coarse_only.py \
  --ckpt_path checkpoints_road_coarse_stage/best.pt
```

Note:  
`checkpoints_road_coarse_stage/best.pt` is not kept in the final clean folder.

---

# 6. Inference interface

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
- `road_condition_direct`
- `road_condition_infer`
- `road_state`
- `road_severity`
- `drivable_summary`

`road_condition_infer` uses the current best road inference logic:
- multi-crop evaluation
- auxiliary reranking with `road_state` and `road_severity`
- crop fusion

## Command-line inference example

Single image:
```bash
python scripts/infer_pipeline_json.py \
  --input_path assets/demo.jpg \
  --output_dir infer_outputs/demo_json
```

Directory inference:
```bash
python scripts/infer_pipeline_json.py \
  --input_path /path/to/images \
  --output_dir infer_outputs/demo_json_dir \
  --max_images 20
```

Recursive directory inference:
```bash
python scripts/infer_pipeline_json.py \
  --input_path /path/to/images \
  --output_dir infer_outputs/demo_json_dir \
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

## Python API example

Single image:
```python
from utils.infer_api import load_pipeline, infer_single_image_path

pipe = load_pipeline(
    ckpt_path="checkpoints_road_coarse_to_fine/best.pt"
)

result = infer_single_image_path(
    image_path="/path/to/demo.jpg",
    pipeline=pipe,
)

print(result["prediction"]["time"]["label"])
print(result["prediction"]["road_condition_infer"]["label"])
```

Directory:
```python
from utils.infer_api import load_pipeline, infer_path, save_results_to_dir

pipe = load_pipeline(
    ckpt_path="checkpoints_road_coarse_to_fine/best.pt"
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

---

# 7. Kept files in the clean folder

This clean folder keeps:
- main code related to the current route
- final mainline checkpoint only
- key cache for RSCD sample weighting

Important kept items:
- `configs/`
- `data/`
- `models/`
- `utils/`
- `scripts/`
- `checkpoints_road_coarse_to_fine/best.pt`
- `cache/hard_weight_cache/rscd_train_sample_weights.pt`

Removed items:
- older experiment checkpoints
- logs
- analysis outputs
- infer outputs
- eval outputs from previous messy versions

---

# 8. Notes

This clean version is intended for:
- final mainline evaluation
- teammate integration
- reusable inference interface
- easier local code reading

It is not intended to preserve every historical experiment artifact.
