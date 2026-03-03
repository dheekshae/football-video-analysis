# ⚽ Football Video Analysis System

A computer vision pipeline for player detection, multi-object tracking, speed estimation, and spatial heatmap generation from football match footage.

Built using YOLOv8 and ByteTrack.

---

## 🚀 Features

- Player detection using YOLOv8n
- Multi-object tracking with ByteTrack
- Stable track filtering (MIN_STEPS threshold)
- Player speed estimation (px/sec)
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


```
football-video-analysis/
│
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── metrics.py
│   ├── visualizer.py
│   ├── video_io.py
│   └── write_annotated_video.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ▶️ Run The Pipeline

```bash
python -m src.write_annotated_video
Outputs

outputs/annotated.avi

outputs/metrics.csv

outputs/heatmaps/

🔬 Technical Highlights

Frame-by-frame inference pipeline

Persistent player ID tracking

Pixel displacement-based speed estimation

Heatmap density accumulation

Stable track filtering via minimum frame threshold

🔮 Future Improvements

Real-world speed calibration (pixel → meters)

Team classification via jersey color clustering

Tactical formation analysis

Distance covered per player

Sprint detection

📌 Author

Dheeksha E
