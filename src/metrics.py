from dataclasses import dataclass, field
from collections import deque
import numpy as np
import cv2

METERS_PER_PIXEL = 0.007377
MAX_REASONABLE_SPEED_KMPH = 40.0
SMOOTHING_WINDOW = 5


@dataclass
class TrackState:
    last_point: tuple[float, float] | None = None
    total_dist_px: float = 0.0
    steps: int = 0
    heatmap: np.ndarray | None = None
    recent_speeds: deque = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    current_speed_kmph: float = 0.0
    speed_sum_kmph: float = 0.0
    speed_count: int = 0
    max_speed_kmph: float = 0.0


def foot_point_xy(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float(y2)


class Metrics:
    def __init__(self, frame_size: tuple[int, int], fps: float, heatmap_scale: float = 0.25):
        w, h = frame_size
        self.fps = fps
        self.scale = heatmap_scale
        self.hm_w = max(1, int(w * heatmap_scale))
        self.hm_h = max(1, int(h * heatmap_scale))
        self.states: dict[int, TrackState] = {}

    def update(self, track_id: int, bbox_xyxy: np.ndarray):
        s = self.states.setdefault(track_id, TrackState())
        px, py = foot_point_xy(bbox_xyxy)

        if s.heatmap is None:
            s.heatmap = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)

        sx, sy = int(px * self.scale), int(py * self.scale)
        if 0 <= sx < self.hm_w and 0 <= sy < self.hm_h:
            s.heatmap[sy, sx] += 1.0

        if s.last_point is not None:
            dx, dy = px - s.last_point[0], py - s.last_point[1]
            dist_px = float((dx * dx + dy * dy) ** 0.5)
            s.total_dist_px += dist_px
            s.steps += 1

            speed_kmph = min(dist_px * METERS_PER_PIXEL * self.fps * 3.6, MAX_REASONABLE_SPEED_KMPH)
            s.recent_speeds.append(speed_kmph)

            smooth_speed = sum(s.recent_speeds) / len(s.recent_speeds)
            s.current_speed_kmph = smooth_speed
            s.speed_sum_kmph += smooth_speed
            s.speed_count += 1
            s.max_speed_kmph = max(s.max_speed_kmph, smooth_speed)

        s.last_point = (px, py)

    def current_speed_kmph(self, track_id: int) -> float:
        s = self.states.get(track_id)
        return round(s.current_speed_kmph, 2) if s else 0.0

    def avg_speed_kmph(self, track_id: int) -> float:
        s = self.states.get(track_id)
        return round(s.speed_sum_kmph / s.speed_count, 2) if s and s.speed_count else 0.0

    def max_speed_kmph(self, track_id: int) -> float:
        s = self.states.get(track_id)
        return round(s.max_speed_kmph, 2) if s else 0.0

    def total_distance_m(self, track_id: int) -> float:
        s = self.states.get(track_id)
        return round(s.total_dist_px * METERS_PER_PIXEL, 2) if s else 0.0

    def frames_tracked(self, track_id: int) -> int:
        s = self.states.get(track_id)
        return s.steps + 1 if s and s.last_point is not None else 0

    def summary_rows(self, min_steps: int = 1) -> list[dict]:
        return [
            {
                "player_id": tid,
                "frames_tracked": self.frames_tracked(tid),
                "avg_speed_kmph": self.avg_speed_kmph(tid),
                "max_speed_kmph": self.max_speed_kmph(tid),
                "total_distance_m": self.total_distance_m(tid),
            }
            for tid, s in sorted(self.states.items())
            if s.steps >= min_steps
        ]

    def save_heatmap(self, track_id: int, out_path: str):
        s = self.states.get(track_id)
        if not s or s.heatmap is None or s.heatmap.max() <= 0:
            return

        hm = cv2.GaussianBlur(s.heatmap, (0, 0), sigmaX=3)
        hm = hm / (hm.max() + 1e-6)
        img = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(out_path, img)