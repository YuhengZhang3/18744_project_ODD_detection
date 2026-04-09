# Unified Evaluation Summary

checkpoint: `checkpoints_anomaly_head/best_model_only.pt`

## BDD

- time acc: 0.8982
- scene acc: 0.7018
- visibility acc: 0.8703
- drivable mIoU all: 0.6744
- drivable mIoU fg: 0.5460
- drivable foreground IoU: 0.4205

## Anomalies

- anomalies acc: 0.9951
- anomaly file: `anomaly/anomaly_eval.json`

## Road

- road direct acc: 0.2778
- road infer-style acc: 0.3083

- road aux file: `road_aux/road_aux_eval.json`
- road relaxed file: `road_relaxed/road_relaxed.txt`