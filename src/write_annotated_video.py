import os
import pandas as pd

from src.video_io import VideoReader, VideoWriter
from src.detector import YoloPersonDetector
from src.tracker import ByteTrackerWrapper
from src.visualizer import draw_tracks
from src.metrics import Metrics


def main():
    input_path = "assets/input.mp4"
    output_path = "outputs/annotated.avi"

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/heatmaps", exist_ok=True)

    reader = VideoReader(input_path)
    writer = VideoWriter(output_path, fps=reader.fps, size=reader.size)

    detector = YoloPersonDetector(model_path="yolov8s.pt", conf=0.35, iou=0.5)
    tracker = ByteTrackerWrapper()
    metrics = Metrics(frame_size=reader.size, fps=reader.fps, heatmap_scale=0.25)

    seen_ids = set()
    frames = 0

    try:
        while True:
            frame = reader.read_one()
            if frame is None:
                break

            boxes, confs = detector.detect(frame)
            tracks = tracker.update(boxes, confs)

            for tid, box in tracks:
                seen_ids.add(tid)
                metrics.update(tid, box)

            # Show current speed in km/h on the annotated frame
            annotated = draw_tracks(frame, tracks, speed_fn=metrics.current_speed_kmph)

            if (annotated.shape[1], annotated.shape[0]) != reader.size:
                raise RuntimeError(
                    f"Frame size mismatch. got={(annotated.shape[1], annotated.shape[0])} expected={reader.size}"
                )

            writer.write(annotated)

            frames += 1
            if frames % 50 == 0:
                print("Processed frames:", frames)

    finally:
        reader.release()
        writer.release()

    # Filter out tiny/unstable tracks
    MIN_STEPS = 25  # ~1 second at 25 fps

    rows = []
    for tid in sorted(seen_ids):
        state = metrics.states.get(tid)
        if state is None or state.steps < MIN_STEPS:
            continue

        rows.append(
            {
                "player_id": tid,
                "frames_tracked": metrics.frames_tracked(tid),
                "avg_speed_kmph": round(metrics.avg_speed_kmph(tid), 2),
                "max_speed_kmph": round(metrics.max_speed_kmph(tid), 2),
                "total_distance_m": round(metrics.total_distance_m(tid), 2),
            }
        )

        metrics.save_heatmap(tid, f"outputs/heatmaps/player_{tid}.png")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"Saved: {output_path}")
        print(f"No tracks met MIN_STEPS={MIN_STEPS}. Try lowering MIN_STEPS.")
        print("\nTotal frames written:", frames)
        return

    df = df.sort_values("player_id")
    df.to_csv("outputs/metrics.csv", index=False)

    print(f"\nSaved: {output_path}")
    print("Saved: outputs/metrics.csv")
    print("Saved: outputs/heatmaps/player_*.png")
    print("\n=== Player Metrics (filtered) ===")
    print(df.to_string(index=False))
    print("\nTotal frames written:", frames)
    print("Tracks reported:", len(df))


if __name__ == "__main__":
    main()