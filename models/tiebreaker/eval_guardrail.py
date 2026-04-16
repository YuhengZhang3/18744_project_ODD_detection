# models/tiebreaker/eval_guardrail.py
"""
Evaluate Guardrail soft-mask quality on the same val split used for MLP training.

No direct GT exists for road_condition(27) or visibility(3), so we use three
indirect proxies:

  1. Consistency check (quantitative):
     Count samples where MLP confidently predicts rain/snow but the final
     road_condition argmax still falls in the 'dry' state. Compare pre- vs
     post-Guardrail counts.

  2. Road-state agreement (semi-quantitative):
     The yuheng ODD model has an auxiliary road_state(5) head trained on RSCD
     with GT. Compare agreement between road_state argmax and the state-prefix
     of road_condition argmax, pre vs post Guardrail.

  3. Qualitative HTML report:
     Sample N examples where Guardrail changed the argmax. Render image +
     pre/post top-3 + MLP weather + 5-state aggregate side-by-side.

The script does NOT modify data_harvester.py or require re-running the .pt file.
It rebuilds the filename list by replaying the harvester's traversal order.

Usage:
    python -m models.tiebreaker.eval_guardrail

    # or with custom paths:
    python -m models.tiebreaker.eval_guardrail \
        --tiebreaker_pt data/tiebreaker_train.pt \
        --mlp_ckpt checkpoints_tiebreaker/tiebreaker_best.pt \
        --out_dir eval_guardrail_output \
        --html_samples 40
"""

import os
import sys
import json
import glob
import argparse
import base64
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tiebreaker.tiebreakers_guard import (
    ROAD_COND_STATES,
    ROAD_COND_NAMES,
    run_guardrail,
)
from models.tiebreaker.tiebreaker_mlp import TiebreakerMLP
from scripts.data_harvester import (
    load_bdd_labels,
    load_acdc_labels,
    load_roadwork_labels,
)


# ============================================================
# State mapping: 27-class argmax -> 5-state index
# 5-state order matches yuheng road_state head: [dry, wet, water, snow, ice]
# ============================================================
ROAD_STATE_ORDER = ["dry", "wet", "water", "snow", "ice"]
ROAD_STATE_NAME_TO_IDX = {s: i for i, s in enumerate(ROAD_STATE_ORDER)}


def road_cond_idx_to_state_idx(idx_27: int) -> int:
    """Map a 27-class road_condition index to its 5-state prefix index."""
    for state_name, idxs in ROAD_COND_STATES.items():
        if idx_27 in idxs:
            return ROAD_STATE_NAME_TO_IDX[state_name]
    raise ValueError(f"Invalid road_condition index: {idx_27}")


# Precompute lookup array for vectorized use
_IDX27_TO_STATE5 = np.array(
    [road_cond_idx_to_state_idx(i) for i in range(27)], dtype=np.int64
)


# ============================================================
# Stage 1: Rebuild sample list in harvester order
# ============================================================
def rebuild_sample_list(args):
    """
    Walk the three datasets in the exact same order as data_harvester.harvest_dataset.
    Returns list of (dataset_name, basename, merged_json_path).

    Must match the row order of tiebreaker_train.pt so that indices from the
    same seed=42 split map to the correct merged JSON.
    """
    samples = []

    def resolve(p):
        return p if os.path.isabs(p) else str(PROJECT_ROOT / p)

    # --- BDD ---
    print("  Loading BDD labels...")
    bdd_labels = load_bdd_labels(resolve(args.bdd_labels))
    bdd_jsons = sorted(glob.glob(os.path.join(resolve(args.bdd_outputs), "*.json")))
    for jf in bdd_jsons:
        basename = os.path.splitext(os.path.basename(jf))[0]
        if basename not in bdd_labels:
            continue
        samples.append(("bdd", basename, jf))

    # --- ACDC ---
    print("  Loading ACDC labels...")
    acdc_labels = load_acdc_labels(resolve(args.acdc_root))
    acdc_jsons = sorted(glob.glob(os.path.join(resolve(args.acdc_outputs), "*.json")))
    for jf in acdc_jsons:
        basename = os.path.splitext(os.path.basename(jf))[0]
        if basename not in acdc_labels:
            continue
        samples.append(("acdc", basename, jf))

    # --- Roadwork ---
    print("  Loading Roadwork labels...")
    rw_label_paths = [resolve(p) for p in args.rw_labels]
    rw_labels = load_roadwork_labels(rw_label_paths)
    rw_jsons = sorted(glob.glob(os.path.join(resolve(args.rw_outputs), "*.json")))
    for jf in rw_jsons:
        basename = os.path.splitext(os.path.basename(jf))[0]
        if basename not in rw_labels:
            continue
        samples.append(("roadwork", basename, jf))

    return samples


def reproduce_val_split(n_total, seed=42, val_ratio=0.15):
    """
    Reproduce the exact same 85/15 split used in train_tiebreaker.py.
    
    torch.utils.data.random_split internally does:
        indices = randperm(sum(lengths), generator=...)
        then slices them in the order of `lengths` argument.
    
    train is passed first, then val, so val is the *second* slice.
    """
    n_val = int(n_total * val_ratio)      # note: int(), not round()
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).numpy()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:n_train + n_val]
    return val_indices, train_indices


# ============================================================
# Stage 2: MLP inference
# ============================================================
def run_mlp_inference(model, X_tensor, device="cpu", batch_size=512):
    """
    Run TiebreakerMLP forward and extract fog/rain/snow sigmoid probabilities.

    ASSUMPTION: TiebreakerMLP.forward returns a dict with at least the keys
    'fog', 'rain', 'snow', each being a [B, 1] or [B] tensor of BCE logits.
    If your forward returns a tuple/list or raw concat tensor, adapt the
    extraction block below.
    """
    model.eval()
    model.to(device)

    n = X_tensor.shape[0]
    fog = np.zeros(n, dtype=np.float32)
    rain = np.zeros(n, dtype=np.float32)
    snow = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_tensor[start:end].to(device)
            out = model(xb)

            # --- Extraction block: adapt if your MLP interface differs ---
            fog_logits = out["fog"].squeeze(-1) if out["fog"].ndim > 1 else out["fog"]
            rain_logits = out["rain"].squeeze(-1) if out["rain"].ndim > 1 else out["rain"]
            snow_logits = out["snow"].squeeze(-1) if out["snow"].ndim > 1 else out["snow"]
            # -------------------------------------------------------------

            fog[start:end] = torch.sigmoid(fog_logits).cpu().numpy()
            rain[start:end] = torch.sigmoid(rain_logits).cpu().numpy()
            snow[start:end] = torch.sigmoid(snow_logits).cpu().numpy()

    return {"fog": fog, "rain": rain, "snow": snow}


# ============================================================
# Stage 3: Load stage-1 softmax + run Guardrail per sample
# ============================================================
def load_stage1_probs(merged_path):
    """
    Extract the three softmax vectors we need from a merged JSON:
      - road_condition_infer (27,)
      - road_state (5,)
      - visibility (3,)
    Also return the image_path for HTML rendering.

    Returns dict or None on missing fields.
    """
    with open(merged_path, "r") as f:
        merged = json.load(f)

    pred = merged.get("yuheng", {}).get("prediction", {})
    if not pred:
        return None

    rc = pred.get("road_condition_infer") or pred.get("road_condition_direct")
    rs = pred.get("road_state")
    vis = pred.get("visibility")

    if not (rc and rs and vis):
        return None

    try:
        road27 = np.array(rc["probabilities"], dtype=np.float32)
        road_state5 = np.array(rs["probabilities"], dtype=np.float32)
        vis3 = np.array(vis["probabilities"], dtype=np.float32)
    except (KeyError, TypeError):
        return None

    if road27.shape != (27,) or road_state5.shape != (5,) or vis3.shape != (3,):
        return None

    return {
        "road27": road27,
        "road_state5": road_state5,
        "vis3": vis3,
        "image_path": merged.get("yuheng", {}).get("image_path", ""),
    }


# ============================================================
# Metrics
# ============================================================
def compute_metrics(pre_road27, post_road27, pre_vis, post_vis,
                    road_state5, mlp_probs, threshold=0.5):
    """
    pre_road27, post_road27: [N, 27]
    pre_vis, post_vis:       [N, 3]
    road_state5:             [N, 5]
    mlp_probs:               dict of {fog, rain, snow}, each [N]
    """
    pre_argmax = pre_road27.argmax(axis=1)
    post_argmax = post_road27.argmax(axis=1)

    pre_states = _IDX27_TO_STATE5[pre_argmax]   # [N]
    post_states = _IDX27_TO_STATE5[post_argmax]
    rs_argmax = road_state5.argmax(axis=1)

    rain_high = mlp_probs["rain"] > threshold
    snow_high = mlp_probs["snow"] > threshold
    fog_high = mlp_probs["fog"] > threshold

    # --- 1. Consistency: MLP asserts weather, but state still 'dry' ---
    dry_idx = ROAD_STATE_NAME_TO_IDX["dry"]
    pre_rain_contra = int(((pre_states == dry_idx) & rain_high).sum())
    post_rain_contra = int(((post_states == dry_idx) & rain_high).sum())
    pre_snow_contra = int(((pre_states == dry_idx) & snow_high).sum())
    post_snow_contra = int(((post_states == dry_idx) & snow_high).sum())

    # --- 2. Road-state agreement ---
    pre_agree = float((pre_states == rs_argmax).mean())
    post_agree = float((post_states == rs_argmax).mean())

    # --- 3. Visibility sanity (fog high -> good visibility should drop) ---
    pre_vis_argmax = pre_vis.argmax(axis=1)   # 0=poor, 1=medium, 2=good
    post_vis_argmax = post_vis.argmax(axis=1)
    # Count samples where MLP says fog high but vis argmax is still 'good' (2)
    pre_vis_contra = int(((pre_vis_argmax == 2) & fog_high).sum())
    post_vis_contra = int(((post_vis_argmax == 2) & fog_high).sum())

    return {
        "n_samples": int(len(pre_argmax)),
        "mlp_threshold": threshold,
        "mlp_high_counts": {
            "fog": int(fog_high.sum()),
            "rain": int(rain_high.sum()),
            "snow": int(snow_high.sum()),
        },
        # --- proxy 1: consistency ---
        "rain_state_contradictions": {
            "pre": pre_rain_contra,
            "post": post_rain_contra,
            "drop": pre_rain_contra - post_rain_contra,
            "drop_pct": (
                100.0 * (pre_rain_contra - post_rain_contra) / max(1, pre_rain_contra)
            ),
        },
        "snow_state_contradictions": {
            "pre": pre_snow_contra,
            "post": post_snow_contra,
            "drop": pre_snow_contra - post_snow_contra,
            "drop_pct": (
                100.0 * (pre_snow_contra - post_snow_contra) / max(1, pre_snow_contra)
            ),
        },
        "fog_vs_good_vis_contradictions": {
            "pre": pre_vis_contra,
            "post": post_vis_contra,
            "drop": pre_vis_contra - post_vis_contra,
        },
        # --- proxy 2: road_state agreement ---
        "road_state_agreement": {
            "pre": pre_agree,
            "post": post_agree,
            "delta": post_agree - pre_agree,
        },
        # --- general ---
        "road_argmax_changed": int((pre_argmax != post_argmax).sum()),
        "vis_argmax_changed": int((pre_vis_argmax != post_vis_argmax).sum()),
    }


def print_metrics(metrics):
    m = metrics
    print("\n" + "=" * 60)
    print("  GUARDRAIL EVALUATION METRICS")
    print("=" * 60)
    print(f"  Samples evaluated:   {m['n_samples']}")
    print(f"  MLP threshold:       {m['mlp_threshold']}")
    print(f"  MLP 'high' counts:   fog={m['mlp_high_counts']['fog']}  "
          f"rain={m['mlp_high_counts']['rain']}  snow={m['mlp_high_counts']['snow']}")
    print()
    print("  [Proxy 1] Weather-vs-state contradictions (want: drop)")
    r = m["rain_state_contradictions"]
    print(f"    rain high  & state=dry:   {r['pre']:5d} -> {r['post']:5d}  "
          f"(-{r['drop']}, -{r['drop_pct']:.1f}%)")
    s = m["snow_state_contradictions"]
    print(f"    snow high  & state=dry:   {s['pre']:5d} -> {s['post']:5d}  "
          f"(-{s['drop']}, -{s['drop_pct']:.1f}%)")
    v = m["fog_vs_good_vis_contradictions"]
    print(f"    fog high   & vis=good:    {v['pre']:5d} -> {v['post']:5d}  (-{v['drop']})")
    print()
    print("  [Proxy 2] road_condition state-prefix vs road_state argmax (want: up)")
    a = m["road_state_agreement"]
    print(f"    agreement: {a['pre']*100:5.2f}% -> {a['post']*100:5.2f}%  "
          f"(delta {a['delta']*100:+.2f}%)")
    print()
    print(f"  road_condition argmax changed by Guardrail: {m['road_argmax_changed']} samples")
    print(f"  visibility argmax changed by Guardrail:      {m['vis_argmax_changed']} samples")
    print("=" * 60)


# ============================================================
# Stage 4: HTML sample report
# ============================================================
def encode_image_b64(path, max_bytes=500_000):
    """Encode an image as base64 data URI. Returns empty string on failure."""
    try:
        if not path or not os.path.exists(path):
            return ""
        size = os.path.getsize(path)
        if size > max_bytes * 4:
            # Skip very large images to keep HTML manageable
            return ""
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else (
            "image/png" if ext == ".png" else "image/jpeg"
        )
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""


def format_topk(probs_27, k=3):
    """Format top-k classes as 'name (p=0.XX)' strings."""
    top = np.argsort(probs_27)[::-1][:k]
    return [(ROAD_COND_NAMES[i], float(probs_27[i])) for i in top]


def aggregate_5state(probs_27):
    """Sum 27-dim softmax by state prefix."""
    out = {}
    for state, idxs in ROAD_COND_STATES.items():
        out[state] = float(probs_27[idxs].sum())
    return out


def generate_html_report(report_samples, out_path):
    """
    report_samples: list of dicts with keys:
      dataset, basename, image_path,
      mlp_fog, mlp_rain, mlp_snow,
      pre_top3, post_top3, pre_agg, post_agg,
      pre_vis, post_vis,
      road_state_probs
    """
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Guardrail Evaluation Report</title>",
        "<style>",
        "body { font-family: sans-serif; background: #1a1a1a; color: #ddd; max-width: 1200px; margin: 20px auto; padding: 20px; }",
        "h1 { color: #fff; }",
        ".sample { border: 1px solid #444; border-radius: 8px; padding: 16px; margin: 16px 0; background: #222; }",
        ".header { font-weight: bold; margin-bottom: 8px; color: #fff; }",
        ".grid { display: grid; grid-template-columns: 300px 1fr 1fr; gap: 16px; }",
        ".img-box img { max-width: 300px; border-radius: 4px; }",
        ".img-box .placeholder { color: #888; padding: 40px; background: #333; text-align: center; border-radius: 4px; }",
        ".col h3 { margin-top: 0; color: #8cf; font-size: 14px; }",
        ".topk { font-family: monospace; font-size: 12px; line-height: 1.5; }",
        ".agg { font-family: monospace; font-size: 12px; margin-top: 6px; }",
        ".bar { display: inline-block; height: 10px; background: #48a; margin-left: 4px; vertical-align: middle; }",
        ".changed { color: #fd6; }",
        ".weather { font-size: 12px; color: #aaa; margin-top: 4px; }",
        "</style></head><body>",
        f"<h1>Guardrail Evaluation Report ({len(report_samples)} samples)</h1>",
        "<p>Samples below had their road_condition argmax changed by Guardrail. "
        "Left: original stage-1 top-3. Right: post-Guardrail top-3. "
        "5-state aggregates shown underneath each.</p>",
    ]

    for s in report_samples:
        pre_top = s["pre_top3"]
        post_top = s["post_top3"]
        pre_agg = s["pre_agg"]
        post_agg = s["post_agg"]

        img_b64 = encode_image_b64(s["image_path"])
        img_html = (
            f"<img src='{img_b64}' alt='{s['basename']}'/>" if img_b64
            else f"<div class='placeholder'>image unavailable<br>{s['image_path']}</div>"
        )

        def topk_block(top):
            return "<br>".join(f"{name:24s} {p:.3f}" for name, p in top)

        def agg_block(agg):
            parts = []
            for state in ROAD_STATE_ORDER:
                v = agg[state]
                parts.append(
                    f"{state:6s} {v:.3f}"
                    f"<span class='bar' style='width:{int(v*120)}px'></span>"
                )
            return "<br>".join(parts)

        # Highlight post_top[0] if it's different from pre_top[0]
        changed_cls = " changed" if pre_top[0][0] != post_top[0][0] else ""

        html.append(f"""
<div class='sample'>
  <div class='header'>[{s['dataset']}] {s['basename']}</div>
  <div class='weather'>
    MLP: fog={s['mlp_fog']:.2f}  rain={s['mlp_rain']:.2f}  snow={s['mlp_snow']:.2f}
    &nbsp;|&nbsp;
    road_state argmax: {s['road_state_name']} ({s['road_state_conf']:.2f})
  </div>
  <div class='grid'>
    <div class='img-box'>{img_html}</div>
    <div class='col'>
      <h3>stage-1 top-3 (pre)</h3>
      <div class='topk'>{topk_block(pre_top)}</div>
      <h3>5-state aggregate (pre)</h3>
      <div class='agg'>{agg_block(pre_agg)}</div>
      <div class='weather'>vis: poor={s['pre_vis'][0]:.2f} med={s['pre_vis'][1]:.2f} good={s['pre_vis'][2]:.2f}</div>
    </div>
    <div class='col'>
      <h3 class='{changed_cls.strip()}'>post-Guardrail top-3</h3>
      <div class='topk'>{topk_block(post_top)}</div>
      <h3>5-state aggregate (post)</h3>
      <div class='agg'>{agg_block(post_agg)}</div>
      <div class='weather'>vis: poor={s['post_vis'][0]:.2f} med={s['post_vis'][1]:.2f} good={s['post_vis'][2]:.2f}</div>
    </div>
  </div>
</div>""")

    html.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(html))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Guardrail on val split")
    parser.add_argument("--tiebreaker_pt", default="data/tiebreaker_train.pt")
    parser.add_argument("--mlp_ckpt", default="checkpoints_tiebreaker/tiebreaker_best.pt")
    parser.add_argument("--bdd_outputs", default="stage1_outputs_BDD/merged_json")
    parser.add_argument("--bdd_labels", default="data/bdd100k_val/labels")
    parser.add_argument("--acdc_outputs", default="stage1_outputs_ACDC/merged_json")
    parser.add_argument("--acdc_root", default="data/ACDC_val")
    parser.add_argument("--rw_outputs", default="stage1_outputs_roadwork/merged_json")
    parser.add_argument("--rw_labels", nargs="+", default=[
        "data/roadwork_main/annotations/instances_train_gps_split.json",
        "data/roadwork_main/annotations/instances_val_gps_split.json",
    ])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="MLP sigmoid threshold for 'high' weather assertion")
    parser.add_argument("--out_dir", default="eval_guardrail_output")
    parser.add_argument("--html_samples", type=int, default=40,
                        help="Max number of changed samples to render in HTML")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ========== Stage 1: Load data & rebuild filename list ==========
    print("\n[1/4] Loading data and rebuilding sample list...")
    pt_path = args.tiebreaker_pt if os.path.isabs(args.tiebreaker_pt) \
        else str(PROJECT_ROOT / args.tiebreaker_pt)
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    X_all, Y_all = data["X"], data["Y"]
    print(f"  Loaded X: {X_all.shape}, Y: {Y_all.shape}")

    samples = rebuild_sample_list(args)
    print(f"  Rebuilt sample list: {len(samples)} entries")

    if len(samples) != X_all.shape[0]:
        print(f"  WARNING: sample count mismatch ({len(samples)} vs {X_all.shape[0]})")
        print("  This usually means harvester's filter logic diverged from what we")
        print("  replayed here. Check that GT files have not changed since .pt was built.")
        print("  Aborting — otherwise metrics will be wrong.")
        return

    val_idx, _ = reproduce_val_split(len(samples), args.seed, args.val_ratio)
    print(f"  Val split (seed={args.seed}, frac={args.val_ratio}): {len(val_idx)} samples")
    print(f"First 5 val indices: {val_idx[:5]}")

    # ========== Stage 2: Run MLP on val ==========
    print("\n[2/4] Running MLP inference on val split...")
    X_val = X_all[val_idx]
    model = TiebreakerMLP()
    ckpt_path = args.mlp_ckpt if os.path.isabs(args.mlp_ckpt) \
        else str(PROJECT_ROOT / args.mlp_ckpt)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and dict-wrapped checkpoints
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    mlp_probs = run_mlp_inference(model, X_val, device=args.device)
    print(f"  MLP probs computed. "
          f"mean rain={mlp_probs['rain'].mean():.3f}  "
          f"mean snow={mlp_probs['snow'].mean():.3f}  "
          f"mean fog={mlp_probs['fog'].mean():.3f}")

    # ========== Stage 3: Per-sample Guardrail + accumulate ==========
    print("\n[3/4] Running Guardrail on each val sample...")
    val_samples = [samples[i] for i in val_idx]

    n = len(val_samples)
    pre_road27 = np.zeros((n, 27), dtype=np.float32)
    post_road27 = np.zeros((n, 27), dtype=np.float32)
    pre_vis = np.zeros((n, 3), dtype=np.float32)
    post_vis = np.zeros((n, 3), dtype=np.float32)
    road_state5 = np.zeros((n, 5), dtype=np.float32)
    image_paths = [""] * n

    skipped = 0
    for i, (ds, basename, jf) in enumerate(val_samples):
        probs = load_stage1_probs(jf)
        if probs is None:
            skipped += 1
            continue
        pre_road27[i] = probs["road27"]
        pre_vis[i] = probs["vis3"]
        road_state5[i] = probs["road_state5"]
        image_paths[i] = probs["image_path"]

        g = run_guardrail(
            road_softmax=probs["road27"],
            vis_softmax=probs["vis3"],
            p_fog=float(mlp_probs["fog"][i]),
            p_rain=float(mlp_probs["rain"][i]),
            p_snow=float(mlp_probs["snow"][i]),
        )
        post_road27[i] = g["road_condition_corrected"]
        post_vis[i] = g["visibility_corrected"]

    if skipped > 0:
        print(f"  Skipped {skipped} samples with missing stage-1 fields")

    # ========== Metrics ==========
    print("\n[4/4] Computing metrics and generating report...")
    # Filter out skipped rows (zero vectors) from metric computation
    valid_mask = pre_road27.sum(axis=1) > 0
    metrics = compute_metrics(
        pre_road27[valid_mask], post_road27[valid_mask],
        pre_vis[valid_mask], post_vis[valid_mask],
        road_state5[valid_mask],
        {k: v[valid_mask] for k, v in mlp_probs.items()},
        threshold=args.threshold,
    )
    print_metrics(metrics)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # ========== HTML report ==========
    pre_argmax = pre_road27.argmax(axis=1)
    post_argmax = post_road27.argmax(axis=1)
    changed_valid = np.where(valid_mask & (pre_argmax != post_argmax))[0]

    # Sample up to html_samples, prefer ones where MLP was confident
    # (rank by max(rain_prob, snow_prob) so we see decisive Guardrail calls)
    conf_score = np.maximum(mlp_probs["rain"], mlp_probs["snow"])
    ranked = changed_valid[np.argsort(-conf_score[changed_valid])]
    selected = ranked[: args.html_samples]

    report_samples = []
    for i in selected:
        ds, basename, _ = val_samples[i]
        rs_argmax = int(road_state5[i].argmax())
        report_samples.append({
            "dataset": ds,
            "basename": basename,
            "image_path": image_paths[i],
            "mlp_fog": float(mlp_probs["fog"][i]),
            "mlp_rain": float(mlp_probs["rain"][i]),
            "mlp_snow": float(mlp_probs["snow"][i]),
            "pre_top3": format_topk(pre_road27[i]),
            "post_top3": format_topk(post_road27[i]),
            "pre_agg": aggregate_5state(pre_road27[i]),
            "post_agg": aggregate_5state(post_road27[i]),
            "pre_vis": pre_vis[i].tolist(),
            "post_vis": post_vis[i].tolist(),
            "road_state_name": ROAD_STATE_ORDER[rs_argmax],
            "road_state_conf": float(road_state5[i][rs_argmax]),
        })

    html_path = os.path.join(args.out_dir, "report.html")
    generate_html_report(report_samples, html_path)
    print(f"  HTML report saved: {html_path}")
    print(f"  ({len(report_samples)} samples rendered out of "
          f"{len(changed_valid)} Guardrail-changed samples)")


if __name__ == "__main__":
    main()