import cv2
import numpy as np


def draw_tracks(frame, tracks: list[tuple[int, np.ndarray]], speed_fn=None):
    for tid, box in tracks:
        x1, y1, x2, y2 = box.astype(int).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID {tid}"
        if speed_fn is not None:
            label += f" | {speed_fn(tid):.1f}px/s"

        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame