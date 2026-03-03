import cv2
from src.video_io import VideoReader
from src.detector import YoloPersonDetector


def main():
    reader = VideoReader("assets/input.mp4")
    frame = reader.read_one()
    fps = reader.fps
    size = reader.size
    reader.release()

    detector = YoloPersonDetector(model_path="yolov8n.pt", conf=0.35, iou=0.5)
    boxes, confs = detector.detect(frame)

    print("FPS:", fps)
    print("Size:", size)
    print("Person detections:", len(boxes))

    # draw boxes on the frame
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{c:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite("outputs/first_frame_detections.jpg", frame)
    print("Saved: outputs/first_frame_detections.jpg")


if __name__ == "__main__":
    main()