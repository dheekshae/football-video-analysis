from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class TrackState:
    last_center: tuple[float, float] | None = None
    total_dist_px: float = 0.0
    steps: int = 0
    heatmap: np.ndarray | None = None


def center_xy(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


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
        cx, cy = center_xy(bbox_xyxy)

        if s.last_center is not None:
            dx = cx - s.last_center[0]
            dy = cy - s.last_center[1]
            s.total_dist_px += float((dx * dx + dy * dy) ** 0.5)
            s.steps += 1
        s.last_center = (cx, cy)

        if s.heatmap is None:
            s.heatmap = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)

        sx = int(cx * self.scale)
        sy = int(cy * self.scale)
        if 0 <= sx < self.hm_w and 0 <= sy < self.hm_h:
            s.heatmap[sy, sx] += 1.0

    def avg_speed_px_per_sec(self, track_id: int) -> float:
        s = self.states.get(track_id)
        if not s or s.steps == 0:
            return 0.0
        return (s.total_dist_px / s.steps) * self.fps

    def save_heatmap(self, track_id: int, out_path: str):
        s = self.states.get(track_id)
        if not s or s.heatmap is None or s.heatmap.max() <= 0:
            return

        hm = s.heatmap.copy()
        hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=3)
        hm = hm / (hm.max() + 1e-6)

        img = (hm * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(out_path, img)