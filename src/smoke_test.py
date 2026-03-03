import cv2
from ultralytics import YOLO

def main():
    print("OpenCV version:", cv2.__version__)
    model = YOLO("yolov8n.pt")  # first run downloads weights
    print("YOLO loaded:", model.model is not None)

if __name__ == "__main__":
    main()