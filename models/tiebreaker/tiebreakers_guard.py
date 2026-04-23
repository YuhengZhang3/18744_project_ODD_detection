"""
Guardrail soft-mask for road_condition (27) and visibility (3).
Use MLP-corrected fog/rain/snow as the confidence heuristics
Only changes logits, can be run without calling model inference once again
"""

import numpy as np
from typing import Dict, Optional

# ------------------------------------------------------------
# 27-class RSCD road_condition: state grouping
# ------------------------------------------------------------
ROAD_COND_STATES = {
    "dry":   list(range(0, 8)),     # 0-7: asphalt x3, concrete x3, gravel, mud
    "snow":  list(range(8, 10)),    # 8-9: fresh_snow, melted_snow
    "water": list(range(10, 18)),   # 10-17
    "wet":   list(range(18, 26)),   # 18-25
    "ice":   [26],
}

# Add floors ONLY TO ORDINARY CASES；water/ice false positive may be fatal, no floor
FLOORED_STATES = {"dry", "wet", "snow"}

ROAD_COND_NAMES = [
    "dry_asphalt_smooth",   "dry_asphalt_slight",   "dry_asphalt_severe",
    "dry_concrete_smooth",  "dry_concrete_slight",  "dry_concrete_severe",
    "dry_gravel",           "dry_mud",
    "fresh_snow",           "melted_snow",
    "water_asphalt_smooth", "water_asphalt_slight", "water_asphalt_severe",
    "water_concrete_smooth","water_concrete_slight","water_concrete_severe",
    "water_gravel",         "water_mud",
    "wet_asphalt_smooth",   "wet_asphalt_slight",   "wet_asphalt_severe",
    "wet_concrete_smooth",  "wet_concrete_slight",  "wet_concrete_severe",
    "wet_gravel",           "wet_mud",
    "ice",
]

# Visibility order (BDD): [poor, moderate, good]
VIS_POOR, VIS_MOD, VIS_GOOD = 0, 1, 2


# ------------------------------------------------------------
# road_condition plausibility
# ------------------------------------------------------------
def build_road_plausibility(
    p_rain: float,
    p_snow: float,
    alpha: float = 0.7,
    beta: float = 0.8,
    gamma: float = 0.7,
    delta: float = 0.5,
    wet_baseline: float = 0.3,
    floor: float = 0.2,
) -> np.ndarray:
    """
    returns (27,) plausibility vectors
    state-level scalar → broadcast to each state's class index。
    """
    state_p = {
        "dry":   (1.0 - alpha * p_rain) * (1.0 - alpha * p_snow),
        "wet":   wet_baseline + beta * p_rain * (1.0 - p_snow),
        "water": gamma * p_rain,
        "snow":  p_snow,
        "ice":   delta * p_snow,
    }

    for s in state_p:
        lo = floor if s in FLOORED_STATES else 0.0
        state_p[s] = float(np.clip(state_p[s], lo, 1.0))

    plaus = np.zeros(27, dtype=np.float32)
    for state, idxs in ROAD_COND_STATES.items():
        # Normalize by class count so states with more sub-classes (e.g., dry with 8)
        # don't dominate the aggregated mass purely due to cardinality.
        plaus[idxs] = state_p[state] / len(idxs)
    return plaus


def apply_road_guardrail(
    road_softmax: np.ndarray,
    plausibility: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """(27,) softmax * (27,) plausibility → 归一化。plausibility 杀光时回退原值。"""
    assert road_softmax.shape == (27,), f"expected (27,), got {road_softmax.shape}"
    assert plausibility.shape == (27,), f"expected (27,), got {plausibility.shape}"

    weighted = road_softmax * plausibility
    total = weighted.sum()
    if total < eps:
        return road_softmax.copy()
    return (weighted / total).astype(np.float32)


# ------------------------------------------------------------
# visibility plausibility
# ------------------------------------------------------------
def build_visibility_plausibility(
    p_fog: float,
    p_rain: float,
    p_snow: float,
    fog_coef: float = 0.6,
    precip_coef: float = 0.3,
    poor_boost: float = 0.5,
    poor_precip_boost: float = 0.2,
) -> np.ndarray:
    """return (3,) [poor, moderate, good] plausibility。fog contributes the most, while only the max between rain/snow gets a secondary contribution"""
    precip = max(p_rain, p_snow)

    poor_w = 1.0 + poor_boost * p_fog + poor_precip_boost * precip
    mod_w  = 1.0
    good_w = (1.0 - fog_coef * p_fog) * (1.0 - precip_coef * precip)
    good_w = max(0.0, good_w)

    return np.array([poor_w, mod_w, good_w], dtype=np.float32)


def apply_visibility_guardrail(
    vis_softmax: np.ndarray,
    plausibility: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    assert vis_softmax.shape == (3,), f"expected (3,), got {vis_softmax.shape}"
    assert plausibility.shape == (3,), f"expected (3,), got {plausibility.shape}"

    weighted = vis_softmax * plausibility
    total = weighted.sum()
    if total < eps:
        return vis_softmax.copy()
    return (weighted / total).astype(np.float32)


# ------------------------------------------------------------
# aggregate 5-state output (just printed in case the road classification can't do fine granularity)
# ------------------------------------------------------------
def aggregate_road_states(road_softmax_27: np.ndarray) -> Dict[str, float]:
    """ state-prefix-based group-sum"""
    return {
        state: float(road_softmax_27[idxs].sum())
        for state, idxs in ROAD_COND_STATES.items()
    }


# ------------------------------------------------------------
# Entry to the guardrail
# ------------------------------------------------------------
def run_guardrail(
    road_softmax: np.ndarray,
    vis_softmax: np.ndarray,
    p_fog: float,
    p_rain: float,
    p_snow: float,
    road_params: Optional[dict] = None,
    vis_params: Optional[dict] = None,
) -> Dict:
    """
    Takes in stage-1 softmax + MLP-corrected sigmoid weather probabilities。
    Returns corrected softmax + 5-state aggregate + plausibility for eval
    """
    road_params = road_params or {}
    vis_params = vis_params or {}

    road_plaus = build_road_plausibility(p_rain, p_snow, **road_params)
    road_corr  = apply_road_guardrail(road_softmax, road_plaus)

    vis_plaus = build_visibility_plausibility(p_fog, p_rain, p_snow, **vis_params)
    vis_corr  = apply_visibility_guardrail(vis_softmax, vis_plaus)

    return {
        "road_condition_corrected":  road_corr,
        "road_condition_aggregated": aggregate_road_states(road_corr),
        "visibility_corrected":      vis_corr,
        "road_plausibility":         road_plaus,
        "visibility_plausibility":   vis_plaus,
    }


# ------------------------------------------------------------
# sanity check
# ------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # observe the effectiveness of plausibiliity
    road_sm = np.ones(27, dtype=np.float32) / 27
    vis_sm  = np.array([0.1, 0.2, 0.7], dtype=np.float32)

    cases = [
        ("clear day",              dict(p_fog=0.05, p_rain=0.05, p_snow=0.05)),
        ("heavy rain",             dict(p_fog=0.10, p_rain=0.90, p_snow=0.05)),
        ("heavy snow + fog",       dict(p_fog=0.80, p_rain=0.10, p_snow=0.80)),
        ("residual snow, no fall", dict(p_fog=0.10, p_rain=0.05, p_snow=0.25)),
        ("all moderate (uncertain)", dict(p_fog=0.40, p_rain=0.40, p_snow=0.40)),
    ]

    for name, kw in cases:
        r = run_guardrail(road_sm, vis_sm, **kw)
        print(f"--- {name}  |  fog={kw['p_fog']} rain={kw['p_rain']} snow={kw['p_snow']} ---")
        print(f"  road agg:   {r['road_condition_aggregated']}")
        print(f"  visibility: {r['visibility_corrected']}  (poor/mod/good)")
        print()