import os
import cv2
import numpy as np
from typing import Optional, Tuple
from app.core.config import settings


class BallDetector:
    """Dedicated Tennis Ball Detector.
    Supports YOLO (COCO class 32 'sports ball' or fine-tuned weights) with
    resilient HSV color-contour fallback for rapid, motion-blurred tennis balls.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = settings.BALL_CONF_THRESHOLD
    ):
        self.conf_threshold = conf_threshold
        self.model = None

        # Resolve model path (check models/ folder or default model name)
        target_model = model_path or os.path.join(settings.MODELS_DIR, settings.YOLO_MODEL_NAME)
        if not os.path.exists(target_model):
            target_model = settings.YOLO_MODEL_NAME

        try:
            from ultralytics import YOLO
            self.model = YOLO(target_model)
        except Exception as e:
            print(f"[BallDetector] YOLO not available ({e}). Fallback detector active.")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detects tennis ball in a single BGR frame.
        Returns: (center_x, center_y, confidence) or None.
        """
        # 1. Primary: YOLO model inference
        if self.model is not None:
            try:
                results = self.model.predict(
                    source=frame,
                    classes=[32],  # sports ball
                    conf=self.conf_threshold,
                    verbose=False
                )
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_idx = int(boxes.conf.argmax())
                    box = boxes.xyxy[best_idx].cpu().numpy()
                    conf = float(boxes.conf[best_idx].cpu().numpy())
                    cx = float((box[0] + box[2]) / 2.0)
                    cy = float((box[1] + box[3]) / 2.0)
                    return cx, cy, conf
            except Exception:
                pass

        # 2. Secondary Fallback: HSV tennis ball color filter (fluorescent yellow-green)
        return self._color_detect(frame)

    def _color_detect(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([28, 80, 80])
        upper_yellow = np.array([65, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < 2000:
                ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
                circularity = area / (np.pi * (radius ** 2)) if radius > 0 else 0
                if circularity > 0.45:
                    return float(cx), float(cy), 0.55

        return None
