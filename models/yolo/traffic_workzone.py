from pathlib import Path
import sys
import os
from typing import Dict, Any, Union, List
from typing import Optional
import json

# Ensure the project root is in sys.path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.yolo_model import YOLOModel
from configs.yolo_config import yolo_config

class TrafficWorkzoneAnalyzer:
    """
    Analyzes an image using a trained YOLO model to determine traffic density
    for cars, pedestrians, and bicycles, and detects the presence of a workzone.
    """

    def __init__(self, model_path: Union[str, Path], device: str = 'cuda',
                 thresholds_path: Optional[Union[str, Path]] = None):
        """
        Initializes the analyzer with a trained YOLO model.

        Args:
            model_path (Union[str, Path]): Path to the trained YOLO weights.
            device (str): The device to run the model on ('cuda' or 'cpu').
            thresholds_path (Optional[Union[str, Path]]): Path to a JSON file containing
                calibration thresholds for density calculation. If not provided,
                a simple area-only threshold (30%) will be used.
        """
        print(f"loading YOLO model from {model_path} on device {device}...")
        self.yolo_model = YOLOModel(model_name=str(model_path), device=device)
        self.class_names = self.yolo_model.get_model_info()['names']
        self.imgsz = yolo_config.get('imgsz', 640)
        print(f"model loaded with classes: {self.class_names}")

        # Define class IDs based on our class names, see scripts/yolo_style_label_converter.py for details
        self.CAR_CLASS_IDS = [self._get_class_id('vehicle')]
        self.PEDESTRIAN_CLASS_IDS = [self._get_class_id('pedestrian')]
        self.BICYCLE_CLASS_IDS = [self._get_class_id('bicycle')]
        self.WORKZONE_CLASS_IDS = [3, 4, 5, 6]
        self.WORKZONE_CLASS_NAMES = [self.class_names[idx] for idx in self.WORKZONE_CLASS_IDS if idx < len(self.class_names)]

        if any(idx is None for idx in self.CAR_CLASS_IDS + self.PEDESTRIAN_CLASS_IDS + self.BICYCLE_CLASS_IDS):
            print("warning: some core traffic classes not found in model.names. check yolo class mappings.")

        print(f"analyzer configured: car_ids={self.CAR_CLASS_IDS}, pedestrian_ids={self.PEDESTRIAN_CLASS_IDS}, bicycle_ids={self.BICYCLE_CLASS_IDS}")
        print(f"workzone_ids={self.WORKZONE_CLASS_IDS} ({self.WORKZONE_CLASS_NAMES})")

        # Load calibration thresholds if provided
        self.thresholds = None
        if thresholds_path is not None:
            with open(thresholds_path, 'r') as f:
                self.thresholds = json.load(f)
            print(f"loaded density thresholds from {thresholds_path}")
        else:
            print("no threshold provided, use default density factors")

        # Alpha for combining count and area (can be made configurable)
        self.alpha = 0.3

    def _get_class_id(self, name: str) -> Optional[int]:
        for idx, class_name in self.class_names.items():
            if class_name == name:
                return idx
        return None

    def analyze_image(self, image_path: Union[str, Path], return_plot: bool = False) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found at {image_path}")

        results = self.yolo_model.predict(source=str(image_path), save=False, imgsz=self.imgsz, verbose=False)

        if not results:
            return {
                'traffic_density': {'car': 0.0, 'pedestrian': 0.0, 'bicycle': 0.0},
                'work_zone': False,
                'raw_detections': []
            }

        img_result = results[0]
        detections = []
        if img_result.boxes:
            for box in img_result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].cpu().numpy().item()
                cls = int(box.cls[0].cpu().numpy().item())
                detections.append({
                    'bbox': [round(x, 2) for x in xyxy],
                    'confidence': round(conf, 4),
                    'class_id': cls,
                    'class_name': self.class_names.get(cls, f"unknown_{cls}")
                })

        img_width, img_height = img_result.orig_shape[1], img_result.orig_shape[0]
        image_area = img_width * img_height

        traffic_density = self._calculate_traffic_density(detections, image_area)
        is_workzone = self._determine_workzone_presence(detections, image_area)

        result = {
            'traffic_density': traffic_density,
            'work_zone': is_workzone,
            'raw_detections': detections
        }

        if return_plot:
            result['_plot_img'] = img_result.plot()

        return result

    def _calculate_traffic_density(self, detections: List[Dict], image_area: float) -> Dict[str, float]:
        """
        Calculates traffic density scores using either calibrated thresholds or simple area ratio.
        """
        # Initialize counters
        counts = {'car': 0, 'pedestrian': 0, 'bicycle': 0}
        areas = {'car': 0.0, 'pedestrian': 0.0, 'bicycle': 0.0}

        for det in detections:
            cls_id = det['class_id']
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            obj_area = (x2 - x1) * (y2 - y1)

            if cls_id in self.CAR_CLASS_IDS or cls_id == 6:
                counts['car'] += 1
                areas['car'] += obj_area
            elif cls_id in self.PEDESTRIAN_CLASS_IDS:
                counts['pedestrian'] += 1
                areas['pedestrian'] += obj_area
            elif cls_id in self.BICYCLE_CLASS_IDS:
                counts['bicycle'] += 1
                areas['bicycle'] += obj_area

        density_scores = {}

        if self.thresholds is not None:
            # Calibrated method: combine count and area
            for category in ['car', 'pedestrian', 'bicycle']:
                th = self.thresholds.get(category, {})
                max_count = th.get('max_count', 30)          # fallback defaults
                max_area_ratio = th.get('max_area_ratio', 0.2)

                count_score = min(counts[category] / max_count, 1.0) if max_count > 0 else 0.0
                area_ratio = areas[category] / image_area if image_area > 0 else 0.0
                area_score = min(area_ratio / max_area_ratio, 1.0) if max_area_ratio > 0 else 0.0

                density_scores[category] = self.alpha * count_score + (1 - self.alpha) * area_score
                # Clamp to [0,1]
                density_scores[category] = max(0.0, min(1.0, density_scores[category]))
        else:
            # Fallback: simple area ratio with fixed threshold (30%)
            max_area_thresh = image_area * 0.30
            for category in ['car', 'pedestrian', 'bicycle']:
                if max_area_thresh > 0:
                    density_scores[category] = min(areas[category] / max_area_thresh, 1.0)
                else:
                    density_scores[category] = 0.0
                density_scores[category] = max(0.0, min(1.0, density_scores[category]))

        return density_scores

    def _determine_workzone_presence(self, detections: List[Dict], image_area: float) -> bool:
        """
        Determines if a workzone is present based on the ROADWork dataset paper rules,
        plus an additional rule for large workzone objects.
        
        New rule: If the total area of all workzone objects > 30% of the image area,
        then it's considered a workzone regardless of other rules.
        """
        workzone_objects = [det for det in detections if det['class_id'] in self.WORKZONE_CLASS_IDS]

        total_workzone_area = 0.0
        for det in workzone_objects:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            total_workzone_area += (x2 - x1) * (y2 - y1)
        
        area_percentage = total_workzone_area / image_area
        # New Rule: If total workzone area exceeds 50% of image area, it's a workzone
        # I added this rule because we need ways to identify large fences that may only be one single instance
        if image_area > 0 and area_percentage >= 0.50:
            return True

        # Original Rule 1: At least three instances
        if len(workzone_objects) < 3:
            return False

        # Original Rule 2: From two unique categories
        unique_categories = {det['class_id'] for det in workzone_objects}
        if len(unique_categories) < 2:
            return False

        # Original Rule 3: Occupy at least 10% of the image area
        if image_area > 0 and area_percentage < 0.10:
            return False

        return True

def process_traffic_workzone(input_dir, json_dir, model_path, thresholds_path=None, vis_dir=None):
    """
    Evaluate all images in input_dir, save outputs in json_dir.
    If vis_dir is provided, also save bbox visualization images.
    """
    os.makedirs(json_dir, exist_ok=True)
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    analyzer = TrafficWorkzoneAnalyzer(model_path=model_path, thresholds_path=thresholds_path)
    
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in image_extensions]
    
    for img_path in image_paths:
        need_plot = vis_dir is not None
        result = analyzer.analyze_image(str(img_path), return_plot=need_plot)

        output = {
            'car_density': result['traffic_density']['car'],
            'pedestrian_density': result['traffic_density']['pedestrian'],
            'bicycle_density': result['traffic_density']['bicycle'],
            'work_zone': result['work_zone']
        }
        json_path = Path(json_dir) / (img_path.stem + '.json')
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        if need_plot and '_plot_img' in result:
            import cv2
            vis_path = Path(vis_dir) / (img_path.stem + '.jpg')
            cv2.imwrite(str(vis_path), result['_plot_img'])

# for local testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test TrafficWorkzoneAnalyzer on a folder of images.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--model_path', type=str,
                        default='../runs/detect/yolo/stage2/weights/best.pt',
                        help='Path to trained YOLO model weights')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='Path to density thresholds JSON file (optional)')
    parser.add_argument('--output_dir', type=str, default='./traffic_output',
                        help='Directory to save output JSON files (default: ./traffic_output)')
    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir).resolve()
    thresholds_path = Path(args.thresholds).resolve() if args.thresholds else None

    print(f"Processing images in {input_dir}")
    print(f"Using model: {model_path}")
    print(f"Output directory: {output_dir}")
    if thresholds_path:
        print(f"Thresholds file: {thresholds_path}")

    process_traffic_workzone(
        input_dir=str(input_dir),
        json_dir=str(output_dir),
        model_path=str(model_path),
        thresholds_path=str(thresholds_path) if thresholds_path else None
    )

    print("Processing complete.")