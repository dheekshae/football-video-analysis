import cv2


class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

    @property
    def size(self) -> tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def read_one(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()

class VideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.out = cv2.VideoWriter(path, fourcc, fps, size)

        if not self.out.isOpened():
            raise RuntimeError(
                f"VideoWriter failed to open. path={path} fps={fps} size={size} fourcc=MJPG"
            )

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()