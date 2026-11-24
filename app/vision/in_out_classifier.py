from typing import Tuple
import numpy as np
from app.vision.court import CourtModel


class InOutClassifier:
    """Classifies tennis ball bounce events as IN or OUT according to ITF rules:
    - If any part of the tennis ball touches or overlaps the boundary line, the ball is IN.
    - Ball radius is approximately 3.3 cm (0.033 m).
    """

    def __init__(self, court_model: CourtModel, doubles: bool = False):
        self.court = court_model
        self.doubles = doubles
        self.half_len = court_model.half_len
        self.half_width = court_model.half_doubles if doubles else court_model.half_width
        self.ball_radius = court_model.ball_radius

    def classify(self, court_coord: Tuple[float, float]) -> Tuple[bool, float]:
        cx, cy = court_coord
        abs_x = abs(cx)
        abs_y = abs(cy)

        dist_x = abs_x - self.half_len
        dist_y = abs_y - self.half_width

        dx_outside = max(0.0, dist_x)
        dy_outside = max(0.0, dist_y)

        if dx_outside > 0 or dy_outside > 0:
            euclidean_outside = float(np.sqrt(dx_outside**2 + dy_outside**2))
            is_in = euclidean_outside <= self.ball_radius
            dist_cm = (euclidean_outside - self.ball_radius) * 100.0
        else:
            is_in = True
            dist_inside = min(self.half_len - abs_x, self.half_width - abs_y)
            dist_cm = -dist_inside * 100.0

        return is_in, round(dist_cm, 1)
