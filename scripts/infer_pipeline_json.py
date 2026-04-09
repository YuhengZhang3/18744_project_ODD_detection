import os
import sys
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.infer_api import load_pipeline, infer_path, save_results_to_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="single image path or image directory")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints_road_coarse_to_fine/best.pt")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")

    parser.add_argument("--alpha_state", type=float, default=0.35)
    parser.add_argument("--beta_severity", type=float, default=0.10)
    parser.add_argument("--gate_threshold", type=float, default=0.60)
    parser.add_argument("--gate_power", type=float, default=1.5)
    parser.add_argument("--min_mix", type=float, default=0.0)

    parser.add_argument("--road_topk", type=int, default=3)
    parser.add_argument("--cls_topk", type=int, default=3)

    args = parser.parse_args()

    pipeline = load_pipeline(
        ckpt_path=args.ckpt_path,
        alpha_state=args.alpha_state,
        beta_severity=args.beta_severity,
        gate_threshold=args.gate_threshold,
        gate_power=args.gate_power,
        min_mix=args.min_mix,
        road_topk=args.road_topk,
        cls_topk=args.cls_topk,
    )

    results = infer_path(
        input_path=args.input_path,
        pipeline=pipeline,
        max_images=args.max_images,
        recursive=args.recursive,
    )

    saved = save_results_to_dir(
        results=results,
        output_dir=args.output_dir,
        pipeline=pipeline,
        input_path=args.input_path,
    )

    print("saved inference jsons to:", saved["output_dir"])
    print("per-image dir:", saved["per_image_dir"])
    print("summary:", saved["summary_json"])


if __name__ == "__main__":
    main()
