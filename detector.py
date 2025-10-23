from typing import List, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    # We'll raise a helpful error in __init__ if needed.

# If you want a mapping of COCO class ids to names, here's a common list (80 classes).
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu", conf_thresh: float = 0.35):
        """
        Initialize YOLOv8 detector.

        model_name: e.g. "yolov8n.pt" (ultralytics will download if needed)
        device: "cpu" or "cuda"
        conf_thresh: confidence threshold to filter detections
        """
        if YOLO is None:
            raise ImportError("`ultralytics` not installed. Install with: pip install ultralytics")
        self.model = YOLO(model_name)
        # Attempt to set device; ultralytics handles fallbacks.
        try:
            self.model.to(device)
        except Exception as e:
            print(f"Error during cpu run: {e}")
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> List[Tuple[int,int,int,int,float,int,str]]:
        """
        Run detection on a single BGR frame (OpenCV format).

        Returns:
            List of tuples: (x1, y1, x2, y2, conf, class_id, class_name)
            Coordinates are ints (pixel coordinates).
        """
        if frame is None:
            return []
        # ultralytics expects RGB images
        rgb = frame[:, :, ::-1]
        results = self.model.predict(rgb, conf=self.conf_thresh, verbose=False)
        if len(results) == 0:
            return []
        res = results[0]
        print("res=", res)
        detections = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for box in res.boxes:
                # box.cls, box.conf, box.xyxy are common attributes
                try:
                    cls = int(box.cls.cpu().numpy())
                    conf = float(box.conf.cpu().numpy())
                    xyxy = box.xyxy.cpu().numpy().flatten()
                except Exception:
                    # fallback if attributes are plain python types/arrays
                    cls = int(box.cls)
                    conf = float(box.conf)
                    xyxy = np.array(box.xyxy).flatten()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                # 0 = person
                # name = COCO_NAMES[cls] if (cls == 0) else str(cls)
                if cls == 0:
                    name = "person"
                else:
                    return []
                detections.append((x1, y1, x2, y2, conf, cls, name))
        return detections

