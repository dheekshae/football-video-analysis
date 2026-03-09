import cv2
import numpy as np


def draw_tracks(frame, tracks, speed_fn=None):
    out = frame.copy()

    for tid, box in tracks:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if speed_fn is not None:
            speed = speed_fn(tid)
            label = f"ID {tid} | {speed:.1f} km/h"
        else:
            label = f"ID {tid}"

        cv2.putText(
            out,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out