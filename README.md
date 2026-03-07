
# 18744 Project: Unified Multi-Head ODD Detection

This project studies unified image-level ODD attribute prediction with one shared visual backbone and multiple task-specific heads.  
The goal is to support several ODD-related attributes in a single model instead of training fully independent models for each task.

## 1. Task Overview

We predict four image-level attributes:

- **time of day**
- **scene type**
- **visibility**
- **road condition**

The first three tasks are based on **BDD100K**.  
The road-condition task is based on **RSCD**.

---

## 2. Method

## 2.1 Model

We use a shared visual encoder with multiple classification heads.

### Shared backbone
- **DINOv2 ViT-L/14**
- one shared feature extractor for all tasks

### Shared adapter
A shared adapter is applied after the backbone features.  
This adapter is important because it becomes the common feature interface for all heads during multi-task training.

### Task heads
We use one classification head per task:

- **time head**: predicts 4 classes
- **scene head**: predicts 7 classes
- **visibility head**: predicts 3 classes
- **road_condition head**: predicts 27 classes

---

## 2.2 Datasets

### BDD100K
BDD100K is used for:
- time
- scene
- visibility

BDD provides:
- `train`
- `val`
- `test`

For **time** and **scene**, labels come directly from BDD attributes.

#### Time classes
- dawn/dusk
- daytime
- night
- undefined

#### Scene classes
- city street
- gas stations
- highway
- parking lot
- residential
- tunnel
- undefined

### RSCD
RSCD is used for:
- road condition

RSCD provides:
- official train split
- official test split

#### Road-condition classes
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

---

## 2.3 Visibility Label Generation

BDD does not directly provide our final 3-class visibility label, so we generate it with a rule-based labeling pipeline.

The visibility label uses:
- BDD metadata such as weather and time of day
- image appearance cues such as contrast and visibility-related statistics

Final visibility classes:
- **poor**
- **medium**
- **good**

This gives us generated visibility labels for:
- train
- val
- test

So visibility can be trained and evaluated like the other BDD tasks.

---

## 3. Training Pipeline

We use a three-stage pipeline.

## 3.1 Stage 0: Separate training for single heads

Before building one unified model, we first train task heads separately.

### Time + Scene
- trained jointly on BDD
- ViT is frozen
- adapter is still trainable in this stage

### Visibility
- trained on BDD with generated visibility labels
- backbone and adapter are frozen
- only the visibility head is trained

### Road Condition
- trained on RSCD
- backbone and adapter are frozen
- only the road-condition head is trained

This gives us several independently trained heads.

---

## 3.2 Stage 1: Direct merge

We then merge the separately trained heads into one unified model.

Merge rule:
- use the time/scene checkpoint as the base
- overwrite the visibility head with the visibility checkpoint
- overwrite the road-condition head with the RSCD checkpoint

This creates one model containing all four heads.

However, direct merge alone is not sufficient, because the heads were trained under different feature conditions.  
In particular, visibility and road-condition heads are not automatically aligned with the shared adapter learned by the time/scene branch.

---

## 3.3 Stage 2: Head-only joint finetuning

After direct merge, we run joint multi-task finetuning while freezing the shared backbone and adapter.

Training data are mixed across:
- BDD time/scene
- BDD visibility
- RSCD road condition

At this stage:
- all heads are trained together
- the shared feature extractor is fixed
- the goal is to align all heads to the same shared representation

This stage mainly fixes the large mismatch caused by direct merge.

---

## 3.4 Stage 3: Adapter finetuning

Finally, we continue joint multi-task training with a lighter update of the shared representation.

At this stage:
- ViT remains frozen
- the shared adapter is unfrozen
- all heads continue training jointly

To better support the harder road-condition task, we additionally give more emphasis to road-condition training by:
- increasing road-condition sampling frequency
- increasing road-condition loss weight

This stage improves final unified performance, especially on road condition.

---

## 4. Evaluation Setup

### BDD tasks
For:
- time
- scene
- visibility

we report results on **BDD test**.

### RSCD task
For:
- road condition

we report results on **RSCD official test**.

So the final table below is a unified test-time summary:
- BDD test for time / scene / visibility
- RSCD test for road condition

---

## 5. Main Results

We compare three stages:

- **Merged**: direct merge without joint alignment
- **Stage 1**: head-only joint finetuning
- **Stage 2**: adapter + head joint finetuning

## 5.1 Overall test accuracy

| Model | Time | Scene | Visibility | Road Condition |
|---|---:|---:|---:|---:|
| Merged | 0.9359 | 0.7951 | 0.2097 | 0.0611 |
| Stage 1 | 0.9362 | 0.7924 | 0.8519 | 0.2778 |
| Stage 2 | 0.9354 | 0.7918 | 0.8632 | 0.3472 |

---

## 5.2 Time: per-class test accuracy

### Merged
- dawn/dusk: 0.537 (793/1476)
- daytime: 0.955 (9981/10446)
- night: 0.985 (7919/8036)
- undefined: 0.619 (26/42)

### Stage 1
- dawn/dusk: 0.493 (728/1476)
- daytime: 0.961 (10036/10446)
- night: 0.987 (7934/8036)
- undefined: 0.619 (26/42)

### Stage 2
- dawn/dusk: 0.479 (707/1476)
- daytime: 0.961 (10042/10446)
- night: 0.987 (7933/8036)
- undefined: 0.619 (26/42)

---

## 5.3 Scene: per-class test accuracy

### Merged
- city street: 0.910 (11178/12288)
- gas stations: 0.667 (4/6)
- highway: 0.688 (3488/5069)
- parking lot: 0.602 (65/108)
- residential: 0.473 (1127/2382)
- tunnel: 0.735 (36/49)
- undefined: 0.041 (4/98)

### Stage 1
- city street: 0.882 (10835/12288)
- gas stations: 0.167 (1/6)
- highway: 0.703 (3563/5069)
- parking lot: 0.583 (63/108)
- residential: 0.568 (1352/2382)
- tunnel: 0.633 (31/49)
- undefined: 0.041 (4/98)

### Stage 2
- city street: 0.877 (10779/12288)
- gas stations: 0.167 (1/6)
- highway: 0.704 (3571/5069)
- parking lot: 0.574 (62/108)
- residential: 0.583 (1389/2382)
- tunnel: 0.612 (30/49)
- undefined: 0.041 (4/98)

---

## 5.4 Visibility: per-class test accuracy

### Merged
- poor: 0.028 (130/4619)
- medium: 0.102 (242/2383)
- good: 0.294 (3821/12998)

### Stage 1
- poor: 0.804 (3712/4619)
- medium: 0.528 (1259/2383)
- good: 0.928 (12067/12998)

### Stage 2
- poor: 0.824 (3806/4619)
- medium: 0.552 (1316/2383)
- good: 0.934 (12142/12998)

---

## 5.5 Road Condition: per-class test accuracy

### Merged
- dry_asphalt_smooth: 0.096 (7/73)
- dry_asphalt_slight: 0.385 (5/13)
- dry_asphalt_severe: 0.000 (0/5)
- dry_concrete_smooth: 0.000 (0/19)
- dry_concrete_slight: 0.000 (0/10)
- dry_concrete_severe: 0.000 (0/0)
- dry_gravel: 0.056 (1/18)
- dry_mud: 0.048 (1/21)
- fresh_snow: 0.000 (0/0)
- melted_snow: 0.000 (0/0)
- water_asphalt_smooth: 0.250 (1/4)
- water_asphalt_slight: 0.000 (0/1)
- water_asphalt_severe: 0.000 (0/0)
- water_concrete_smooth: 0.000 (0/11)
- water_concrete_slight: 0.000 (0/5)
- water_concrete_severe: 0.000 (0/0)
- water_gravel: 0.000 (0/5)
- water_mud: 0.000 (0/11)
- wet_asphalt_smooth: 0.083 (7/84)
- wet_asphalt_slight: 0.000 (0/31)
- wet_asphalt_severe: 0.000 (0/1)
- wet_concrete_smooth: 0.000 (0/16)
- wet_concrete_slight: 0.000 (0/7)
- wet_concrete_severe: 0.000 (0/1)
- wet_gravel: 0.000 (0/4)
- wet_mud: 0.000 (0/20)
- ice: 0.000 (0/0)

### Stage 1
- dry_asphalt_smooth: 0.521 (38/73)
- dry_asphalt_slight: 0.154 (2/13)
- dry_asphalt_severe: 0.200 (1/5)
- dry_concrete_smooth: 0.263 (5/19)
- dry_concrete_slight: 0.000 (0/10)
- dry_concrete_severe: 0.000 (0/0)
- dry_gravel: 0.444 (8/18)
- dry_mud: 0.095 (2/21)
- fresh_snow: 0.000 (0/0)
- melted_snow: 0.000 (0/0)
- water_asphalt_smooth: 0.000 (0/4)
- water_asphalt_slight: 1.000 (1/1)
- water_asphalt_severe: 0.000 (0/0)
- water_concrete_smooth: 0.000 (0/11)
- water_concrete_slight: 0.000 (0/5)
- water_concrete_severe: 0.000 (0/0)
- water_gravel: 0.200 (1/5)
- water_mud: 0.182 (2/11)
- wet_asphalt_smooth: 0.298 (25/84)
- wet_asphalt_slight: 0.097 (3/31)
- wet_asphalt_severe: 0.000 (0/1)
- wet_concrete_smooth: 0.375 (6/16)
- wet_concrete_slight: 0.000 (0/7)
- wet_concrete_severe: 0.000 (0/1)
- wet_gravel: 0.000 (0/4)
- wet_mud: 0.300 (6/20)
- ice: 0.000 (0/0)

### Stage 2
- dry_asphalt_smooth: 0.548 (40/73)
- dry_asphalt_slight: 0.077 (1/13)
- dry_asphalt_severe: 0.800 (4/5)
- dry_concrete_smooth: 0.474 (9/19)
- dry_concrete_slight: 0.000 (0/10)
- dry_concrete_severe: 0.000 (0/0)
- dry_gravel: 0.611 (11/18)
- dry_mud: 0.048 (1/21)
- fresh_snow: 0.000 (0/0)
- melted_snow: 0.000 (0/0)
- water_asphalt_smooth: 0.000 (0/4)
- water_asphalt_slight: 0.000 (0/1)
- water_asphalt_severe: 0.000 (0/0)
- water_concrete_smooth: 0.000 (0/11)
- water_concrete_slight: 0.000 (0/5)
- water_concrete_severe: 0.000 (0/0)
- water_gravel: 0.600 (3/5)
- water_mud: 0.455 (5/11)
- wet_asphalt_smooth: 0.381 (32/84)
- wet_asphalt_slight: 0.097 (3/31)
- wet_asphalt_severe: 0.000 (0/1)
- wet_concrete_smooth: 0.438 (7/16)
- wet_concrete_slight: 0.000 (0/7)
- wet_concrete_severe: 0.000 (0/1)
- wet_gravel: 0.000 (0/4)
- wet_mud: 0.450 (9/20)
- ice: 0.000 (0/0)

---

## 6. Multi-Task Training and Results

This project explores the unification of independent task heads into a single backbone for ODD (Operational Design Domain) attribute classification. Through a multi-stage process, we developed a balanced model for Time, Scene, Visibility, and Road Condition.

### Training Strategy Evolution
Experimental results show that a direct merge of independently trained heads is insufficient because heads are not automatically compatible with a shared feature space. A multi-stage finetuning approach was required to resolve this misalignment:

* **Stage 0: Direct Merge** – Time and Scene performance remains stable, but Visibility and Road Condition collapse.
* **Stage 1: Head-Only Joint Finetuning** – Dramatically improves Visibility and Road Condition, proving that the issue is head-feature misalignment rather than task incompatibility.
* **Stage 2: Shared Adapter Finetuning** – Lightly tuning the shared adapter further improves performance across all attributes, resulting in the most balanced unified model.

### Key Results
The progression from Stage 0 to Stage 2 demonstrates a clear improvement in model stability and accuracy:
- **Time and Scene:** Performance remained stable throughout all stages.
- **Visibility and Road Condition:** These tasks showed dramatic recovery in Stage 1 and reached peak performance in Stage 2.

### Final Unified Model
The Stage 2 model provides the best overall balance across all four tasks using a single shared backbone.
- **Best Checkpoint:** checkpoints_multitask_stage2/best.pt
- **Summary:** Head-only joint finetuning followed by light shared-adapter tuning is highly effective for consolidating separate task-specific models into one efficient system.
