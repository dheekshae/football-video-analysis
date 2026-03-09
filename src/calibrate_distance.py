import cv2
import math

VIDEO_PATH = "assets/input.mp4"   # change if needed
FRAME_NUMBER = 0                  # choose frame to inspect

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked: ({x}, {y})")


def get_frame(video_path: str, frame_number: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_number}")

    return frame


def main():
    global clicked_points

    frame = get_frame(VIDEO_PATH, FRAME_NUMBER)
    display = frame.copy()

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    print("Instructions:")
    print("1. Click exactly TWO points on the frame.")
    print("2. Press 'q' after clicking.")
    print("3. Use points with known real-world distance.")

    while True:
        temp = display.copy()

        for idx, (x, y) in enumerate(clicked_points):
            cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                temp,
                f"P{idx+1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if len(clicked_points) == 2:
            cv2.line(temp, clicked_points[0], clicked_points[1], (255, 0, 0), 2)

        cv2.imshow("Calibration", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            clicked_points = []
            print("Points reset.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(clicked_points) != 2:
        print("You must click exactly 2 points.")
        return

    (x1, y1), (x2, y2) = clicked_points
    pixel_distance = math.hypot(x2 - x1, y2 - y1)

    print(f"\nPixel distance = {pixel_distance:.2f} pixels")

    real_distance_m = float(input("Enter the real-world distance in meters: "))
    meters_per_pixel = real_distance_m / pixel_distance

    print(f"Meters per pixel = {meters_per_pixel:.6f}")
    print("\nSave this value. We will use it in your metrics pipeline.")


if __name__ == "__main__":
    main()