import numpy as np
import supervision as sv


class ByteTrackerWrapper:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, boxes: np.ndarray, confidences: np.ndarray):
        """
        boxes: (N,4) xyxy
        confidences: (N,)
        returns: list of (track_id, bbox_xyxy)
        """
        if len(boxes) == 0:
            return []

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=np.zeros(len(boxes), dtype=int)
        )

        tracked = self.tracker.update_with_detections(detections)

        results = []
        if tracked.tracker_id is None:
            return []

        for tid, box in zip(tracked.tracker_id, tracked.xyxy):
            results.append((int(tid), box.astype(np.float32)))

        return results