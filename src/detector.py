import cv2
import numpy as np
from ultralytics import YOLO

class SubjectDetector:
    """
    Detects the primary visual subject using YOLOv8 to ensure 
    key content is preserved during reframing.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        # Automatically downloads lightweight YOLOv8 weights on first run
        self.model = YOLO(model_path)

    def detect_primary_subject(self, image: np.ndarray) -> tuple:
        """
        Calculates bounding box [x1, y1, x2, y2] for the largest subject.
        Falls back to center 80% bounding box if no object is detected.
        """
        results = self.model(image, verbose=False)[0]
        boxes = results.boxes

        if len(boxes) == 0:
            h, w, _ = image.shape
            return int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)

        max_area = 0
        best_box = (0, 0, 0, 0)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

        return best_box