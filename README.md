# ⚽ Football Video Analysis System

## 🎥 Demo

![Demo](demo.gif)

A computer vision pipeline for player detection, multi-object tracking, calibrated speed estimation, and spatial heatmap generation from football match footage.

Built using YOLOv8 and ByteTrack.

---

## 🚀 Features

- Player detection using YOLOv8
- Multi-object tracking with ByteTrack
- Stable track filtering for noisy short-lived detections
- Calibrated player speed estimation in km/h
- Per-player total distance covered in meters
- Per-player maximum and average speed metrics
- Per-player heatmap generation
- Annotated video export
- Structured metrics export (CSV)

---

## 🛠 Tech Stack

- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- NumPy
- Pandas
- Matplotlib

---

## 📂 Project Structure

```text
football-video-analysis/
│
├── src/
│   ├── calibrate_distance.py
│   ├── detector.py
│   ├── tracker.py
│   ├── metrics.py
│   ├── visualizer.py
│   ├── video_io.py
│   └── write_annotated_video.py
│
├── assets/
├── outputs/
├── requirements.txt
├── .gitignore
├── demo.gif
└── README.md
▶️ Run The Pipeline
python -m src.write_annotated_video
Outputs

outputs/annotated.avi

outputs/metrics.csv

outputs/heatmaps/

📊 Metrics Export

The generated CSV currently includes:

player_id

frames_tracked

avg_speed_kmph

max_speed_kmph

total_distance_m

🔬 Technical Highlights

Frame-by-frame player detection and tracking pipeline

Foot-point based motion estimation for more stable ground movement tracking

Pixel-to-meter calibration for real-world speed conversion

Smoothed speed estimation to reduce tracking jitter and unrealistic spikes

Per-player movement heatmap generation

📏 Calibration

Player speeds are converted from pixel displacement to real-world units using a manually measured calibration factor:

meters_per_pixel = 0.007377

This enables the pipeline to report player speed in km/h and distance in meters instead of pixel-based units.

🔮 Future Improvements

Speed zone analysis and sprint detection

Team classification via jersey color clustering

Tactical formation and compactness analysis

Homography-based top-down pitch projection

Ball tracking and pass analysis

📌 Author

Dheeksha E
