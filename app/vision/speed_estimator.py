import numpy as np
from typing import List, Tuple


class SpeedEstimator:
    """Calculates ball velocity in km/h using euclidean distance between
    consecutive points on the metric court plane and video FPS.
    """

    def __init__(self, fps: float = 30.0, smoothing_window: int = 3):
        self.fps = fps if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.smoothing_window = smoothing_window
        self.speed_history: List[float] = []

    def calculate_instantaneous_speed(
        self,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float],
        frame_interval: int = 1
    ) -> float:
        if frame_interval <= 0:
            return 0.0

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist_meters = np.sqrt(dx * dx + dy * dy)

        time_delta = frame_interval * self.dt
        if time_delta <= 0:
            return 0.0

        speed_mps = dist_meters / time_delta
        speed_kmh = speed_mps * 3.6

        # Cap realistic tennis ball maximum speed (record ~263 km/h)
        speed_kmh = min(speed_kmh, 280.0)
        return float(speed_kmh)

    def add_speed_sample(self, speed_kmh: float) -> float:
        if speed_kmh > 0:
            self.speed_history.append(speed_kmh)
            if len(self.speed_history) > self.smoothing_window:
                self.speed_history.pop(0)

        if not self.speed_history:
            return 0.0
        return float(np.mean(self.speed_history))
