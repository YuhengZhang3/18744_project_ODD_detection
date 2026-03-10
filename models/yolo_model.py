from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

class YOLOModel:
    """
    Encapsulates a YOLO model by Ultralytics
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', device: str = 'cuda'):
        """
        Args:
            model_name (str): pretrained model name or path e.g. 'yolov8n.pt'。
            device (str): 'cuda' or 'cpu'。
        """
        self.device = device
        self.model = YOLO(model_name)
        self.model.to(device)
        
    def train(self, data: str, epochs: int, batch: int = 16, imgsz: int = 640,
              freeze: Optional[int] = None, lr0: float = 0.001, 
              project: str = 'runs/train', name: str = 'exp', 
              resume: bool = False, **kwargs) -> None:
        """
        Args:
            data (str): path of dataset configuration file data.yaml。
            epochs (int)
            batch (int)
            imgsz (int)
            freeze (Optional[int]): how many layers of backbone to freeze, None if no freeze
            lr0 (float):
            project (str): root dir
            name (str)
            resume (bool): whether to resume from last checkpoint
            **kwargs: other hyperparams supported byu Ultralytics
        """

        train_args = {
            'data': data,
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'lr0': lr0,
            'project': project,
            'name': name,
            'resume': resume,
            'device': self.device,
            **kwargs
        }

        if freeze is not None:
            train_args['freeze'] = freeze
        
        self.model.train(**train_args)
    
    def predict(self, source: Union[str, Path, List[str]], conf: float = 0.25, 
                iou: float = 0.45, imgsz: int = 640, save: bool = False, 
                **kwargs) -> List:
        """
        Args:
            source: Path to image
            conf
            iou: NMS threshold for IoU
            imgsz
            save: whether to save predicted image
            **kwargs
        
        Returns:
            List[ultralytics.engine.results.Results]
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            device=self.device,
            **kwargs
        )
        return results
    
    def export(self, format: str = 'onnx', imgsz: int = 640, **kwargs) -> str:
        """
        Args:
            format (str): format such as 'onnx', 'engine' (TensorRT), 'torchscript'
            imgsz (int)
            **kwargs
        
        Returns:
            str: path to exported file
        """
        export_path = self.model.export(format=format, imgsz=imgsz, **kwargs)
        return export_path
    
    def load_checkpoint(self, weights: str) -> None:
        """
        Args:
            weights (str)
        """
        self.model = YOLO(weights)
        self.model.to(self.device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns:
            dict: model info
        """
        return {
            'names': self.model.names,
            'nc': len(self.model.names),
            'device': str(self.model.device),
            'stride': self.model.stride,
        }
    
    def postprocess_for_odd(self, results: List, img_shape: tuple) -> Dict:
        """
        Args:
            results: list of YOLO prediction results, one per image
            img_shape: image size W,H
        
        Returns:
            dict: indicates traffic density and if work zone is present
        """
        # placeholder for later processing our predictions into ODD tuple format
        # TODO: potentially traversing through all results, linearly interpolate the density of each image
        # compute the amount of workzone objects, use pre-defined hard rules to determine workzone presence
        odd_output = {
            'traffic_density': {
                'car': 0.0,
                'pedestrian': 0.0,
                'bicycle': 0.0
            },
            'work_zone': False
        }
        return odd_output