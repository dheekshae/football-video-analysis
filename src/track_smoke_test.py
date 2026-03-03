from src.video_io import VideoReader
from src.detector import YoloPersonDetector
from src.tracker import ByteTrackerWrapper


def main():
    reader = VideoReader("assets/input.mp4")
    detector = YoloPersonDetector(model_path="yolov8n.pt", conf=0.35, iou=0.5)
    tracker = ByteTrackerWrapper()

    seen = set()
    max_frames = 100
    i = 0

    try:
        for _ in range(max_frames):
            frame = reader.read_one()
            if frame is None:
                break

            boxes, confs = detector.detect(frame)
            tracks = tracker.update(boxes, confs)

            for tid, _ in tracks:
                seen.add(tid)

            if i % 10 == 0:
                print(f"frame={i:03d} detections={len(boxes):02d} tracks={len(tracks):02d} unique_ids={len(seen):02d}")
            i += 1
    finally:
        reader.release()

    print("\nDone.")
    print("Total frames processed:", i)
    print("Unique track IDs seen:", len(seen))


if __name__ == "__main__":
    main()