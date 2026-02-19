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
