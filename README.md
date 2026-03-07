# 18744 Project - Multi-Head ODD Detection

This repository contains our 18-744 project code for image-level ODD attribute prediction with a shared visual backbone and multiple task-specific heads.

## 1. Overview

We build a unified multi-head model for several ODD-related attributes:

- time of day
- scene type
- visibility
- road condition

The model uses one shared image encoder and one shared adapter, then predicts each attribute with an independent classification head.

Our final pipeline has three stages:

1. train single-task heads separately
2. merge the trained heads into one unified model
3. run joint multi-task finetuning to align all heads in the shared feature space

---

## 2. Model Structure

### Backbone
- DINOv2 ViT-L/14
- image input size: `336 x 336`
- output features:
  - CLS token feature
  - patch token features

### Shared adapter
A shared MLP adapter is applied on top of DINOv2 features.

Current implementation:
- shared across all heads
- used to map backbone features to the task feature space

### Heads
Current active heads in code:

- `time`
- `scene`
- `visibility`
- `road_condition`

In the current implementation, these heads are simple MLP heads on top of the shared global feature.

Main related files:
- `models/backbone_dinov2.py`
- `models/heads.py`
- `models/odd_model.py`
- `configs/odd_config.py`

---

## 3. Datasets and Labels

## 3.1 BDD100K

We use BDD100K image-level attributes for three heads:

- `time`
- `scene`
- `visibility`

### Data split
BDD100K provides:
- `train`
- `val`
- `test`

In our current experiments:
- training uses `train`
- model selection mainly uses `val`
- final reporting can use `test`

### Time head
Source label:
- BDD attribute `timeofday`

Classes:
- dawn/dusk
- daytime
- night
- undefined

### Scene head
Source label:
- BDD attribute `scene`

Classes:
- city street
- gas stations
- highway
- parking lot
- residential
- tunnel
- undefined

### Visibility head
Visibility is not directly provided as an original BDD class.  
We generate an additional 3-class label from image statistics and BDD attributes.

Classes:
- poor
- medium
- good

Visibility label files are stored under:
- `bdd100k/visibility_labels/train`
- `bdd100k/visibility_labels/val`
- `bdd100k/visibility_labels/test`

Main related files:
- `data/bdd_dataset.py`
- `scripts/label_visibility_bdd.py`

---

## 3.2 RSCD

We use RSCD for the `road_condition` head.

### Data split
RSCD provides:
- `train-set`
- `test-set`

In training:
- we train on RSCD train
- for stage1/stage2 monitoring, we split RSCD train into train/val internally
- final reporting uses RSCD official test

### Road condition head
The current implementation uses 27 classes:

- dry_asphalt_smooth
- dry_asphalt_slight
- dry_asphalt_severe
- dry_concrete_smooth
- dry_concrete_slight
- dry_concrete_severe
- dry_gravel
- dry_mud
- fresh_snow
- melted_snow
- water_asphalt_smooth
- water_asphalt_slight
- water_asphalt_severe
- water_concrete_smooth
- water_concrete_slight
- water_concrete_severe
- water_gravel
- water_mud
- wet_asphalt_smooth
- wet_asphalt_slight
- wet_asphalt_severe
- wet_concrete_smooth
- wet_concrete_slight
- wet_concrete_severe
- wet_gravel
- wet_mud
- ice

Main related files:
- `data/rscd_dataset.py`
- `scripts/train_road_condition_head.py`
- `scripts/eval_road_condition_head.py`

---

## 4. Visibility Labeling Method

We build visibility labels using BDD attributes and simple image statistics.

### Inputs
From BDD JSON:
- `weather`
- `timeofday`

From image statistics:
- near-road contrast
- far-road contrast
- sky/top-region contrast
- blur / haze related scores

### Output label
We generate:
- `0 = poor`
- `1 = medium`
- `2 = good`

### Rule summary
In general:
- low contrast and fog-like conditions are labeled as poor
- night scenes with partial visibility are labeled as medium
- clear daytime scenes with strong near/far contrast are labeled as good

This is a rule-based pseudo-labeling method used to create a train/val/test label set for the visibility head.

Main scripts:
- `scripts/label_visibility_bdd.py`
- some older visualization / sanity-check scripts were used during development

---

## 5. Training Pipeline

## 5.1 Single-head training

We first trained several heads separately.

### Time + Scene
- dataset: BDD100K
- split: train / val
- script: `scripts/train_time_scene_joint.py`

Important detail:
- ViT backbone is frozen
- adapter is updated in this stage
- `time` and `scene` heads are trained jointly

### Visibility
- dataset: BDD100K with generated visibility labels
- split: train / val
- script: `scripts/train_visibility_head.py`

Important detail:
- backbone and adapter are frozen
- only the `visibility` head is trained

### Road condition
- dataset: RSCD
- split: official train / test
- script: `scripts/train_road_condition_head.py`

Important detail:
- backbone and adapter are frozen
- only the `road_condition` head is trained

---

## 5.2 Merge stage

After separate training, we merge the trained heads into one model.

Merge rule:
- use `time_scene` checkpoint as base
- overwrite `visibility` head from visibility checkpoint
- overwrite `road_condition` head from RSCD checkpoint

Script:
- `scripts/merge_heads.py`

Merged checkpoint:
- `checkpoints_merged/odd_merged_heads.pt`

Why this is needed:
- it combines all trained heads into one shared model
- but direct merge alone does not fully align all heads because they were trained under different feature conditions

---

## 5.3 Stage 1 joint finetuning

Goal:
- align all heads in one shared feature space

Training strategy:
- load merged checkpoint
- freeze full backbone and adapter
- train all heads jointly
- alternate batches from:
  - BDD time/scene
  - BDD visibility
  - RSCD road condition

Script:
- `scripts/train_joint_multidata_stage1.py`

This stage mainly fixes head-feature mismatch after direct merge.

---

## 5.4 Stage 2 joint finetuning

Goal:
- further improve unified performance, especially for road condition

Training strategy:
- initialize from stage1 best checkpoint
- keep ViT frozen
- unfreeze shared adapter
- train adapter + all heads jointly
- increase road-condition training emphasis with:
  - higher road sampling ratio
  - larger road loss weight

Script:
- `scripts/train_joint_multidata_stage2.py`

This stage gives better shared adaptation across tasks.

---

## 6. Evaluation Scripts

Main evaluation scripts:

- `scripts/eval_time_scene.py`
- `scripts/eval_visibility_bdd.py`
- `scripts/eval_road_condition_head.py`
- `scripts/eval_road_condition_valsplit.py`
- `scripts/eval_all_heads.py`

`eval_all_heads.py` can be used to evaluate all heads together on:
- BDD `val` or `test`
- RSCD `test` or internal `valsplit`

---

## 7. Main Results

Below are the current unified-model test results for three stages:

- merged
- stage1
- stage2

### 7.1 Overall test accuracy

| model | time | scene | visibility | road_condition |
|---|---:|---:|---:|---:|
| merged | 0.9359 | 0.7951 | 0.2097 | 0.0611 |
| stage1 | 0.9362 | 0.7924 | 0.8519 | 0.2778 |
| stage2 | 0.9354 | 0.7918 | 0.8632 | 0.3472 |

### 7.2 Result summary

#### Merged
Direct head merging preserves:
- time
- scene

But visibility and road-condition performance collapse after direct merge, which shows that separately trained heads are not automatically aligned in the shared feature space.

#### Stage 1
Stage 1 recovers most of the lost performance:
- time and scene remain stable
- visibility improves strongly
- road_condition improves clearly

This shows that head-only joint finetuning is enough to fix most of the feature mismatch.

#### Stage 2
Stage 2 gives the best final unified model:
- time remains stable
- scene changes only slightly
- visibility improves further
- road_condition improves further

This indicates that lightly adapting the shared adapter helps the model better support all tasks together.

---

## 8. Current Conclusion

Our current experiments support the following conclusion:

- directly merging independently trained heads is not enough
- joint multi-task finetuning is necessary
- stage1 fixes most of the head-feature mismatch
- stage2 further improves unified performance, especially on road_condition
- the final stage2 model is our current best unified multi-head ODD model

Current best checkpoint:
- `checkpoints_multitask_stage2/best.pt`

---

## 9. Main Files

### Models
- `models/backbone_dinov2.py`
- `models/heads.py`
- `models/odd_model.py`

### Datasets
- `data/bdd_dataset.py`
- `data/rscd_dataset.py`

### Single-task training
- `scripts/train_time_scene_joint.py`
- `scripts/train_visibility_head.py`
- `scripts/train_road_condition_head.py`

### Multi-stage unified training
- `scripts/merge_heads.py`
- `scripts/train_joint_multidata_stage1.py`
- `scripts/train_joint_multidata_stage2.py`

### Evaluation
- `scripts/eval_time_scene.py`
- `scripts/eval_visibility_bdd.py`
- `scripts/eval_road_condition_head.py`
- `scripts/eval_road_condition_valsplit.py`
- `scripts/eval_all_heads.py`

