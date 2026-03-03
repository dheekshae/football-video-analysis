from src.video_io import VideoReader


def main():
    reader = VideoReader("assets/input.mp4")
    frame = reader.read_one()

    print("FPS:", reader.fps)
    print("Size (w,h):", reader.size)
    print("First frame is None?", frame is None)

    reader.release()


if __name__ == "__main__":
    main()