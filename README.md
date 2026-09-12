<div align="center">

# AeroNet — Real-Time Drone & Bird Defense Zone Surveillance

**An embedded edge-AI surveillance system combining YOLOv8 target localization, ResNet-18 visual feature extraction, and real-time classification to secure sensitive airspace.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-00FFFF.svg?logo=yolo)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg?logo=render)](https://render.com/)

### 🔗 Live: https://aeronet-ng9t.onrender.com/

</div>

---

## What it is

I built AeroNet for distinguishing between low-flying unmanned aerial vehicles (drones) and natural birds in real time is one of the hardest challenges in perimeter security and airspace monitoring. Traditional radar struggles with small radar cross-sections at low altitudes, while naive computer vision models either choke on latency or misclassify birds as aerial threats.

AeroNet solves this with an edge-deployable, dual-stage vision architecture. It captures live camera feeds, localizes candidate airborne objects in real-time with YOLOv8 at ~15ms inference latency, extracts deep 512-dimensional visual embeddings using a ResNet-18 backbone, and classifies targets with a calibrated classifier — achieving 98.66% validation accuracy. When an unauthorized drone enters the sector, the system highlights the target reticle in red and updates real-time telemetry metrics.

---

## Features

**Dual-Stage Vision Pipeline** — decoupled localization and classification. YOLOv8 isolates regions of interest at 30+ FPS, while ResNet-18 deep feature extraction provides industrial-grade classification accuracy.

**Real-Time Webcam Streaming** — mirrored live video feed with low-latency client-server frame streaming, automatic canvas overlay rendering, and real-time FPS measurement.

**High-Precision Classification** — trained on thousands of aerial drone and bird imagery samples, achieving 98.66% validation accuracy with sub-percent false positive rates.

**Session Telemetry** — tracks live scan count, threat detection ratios, running confidence averages, and real-time audit logs without persistent database clutter.

---

## Model Performance

| Metric | Score |
|---|---|
| Validation Accuracy | **98.66%** |
| Feature Backbone | ResNet-18 (512-dimensional ImageNet embeddings) |
| Classification Head | L2-Regularized Logistic Classifier |
| YOLOv8 Localization Speed | ~15–20 ms per frame |
| Bird Class Accuracy | 98.45% (317 / 322 correctly identified) |
| Drone Class Accuracy | 98.80% (494 / 500 correctly identified) |

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | HTML5 Canvas, Vanilla ES6+, Modern CSS3 Glassmorphism |
| Backend | Python 3.10+, FastAPI, Starlette, Uvicorn |
| Machine Learning | PyTorch, Torchvision (ResNet-18), Scikit-Learn |
| Object Detection | Ultralytics YOLOv8 (Nano) |
| Image Processing | Pillow (PIL), OpenCV Headless (cv2) |
| Cloud / Deployment | Render (Linux Web Service) |

---

## Project structure

```text
AeroNet/
├── artifacts/
│   ├── drone_bird_feature_lr.pkl   # trained ResNet-18 feature classifier bundle
│   ├── metrics.json                # validation metrics and confusion matrix
│   └── prediction_history.json     # runtime session history
├── templates/
│   └── index.html                  # single-page cyberpunk surveillance UI & live stream
├── samples/
│   ├── bird.png                    # test bird sample image
│   └── drone.png                   # test drone sample image
├── app.py                          # FastAPI application and live API endpoints
├── predict.py                      # ResNet-18 feature extraction & inference logic
├── train.py                        # standalone model training pipeline
├── train.ipynb                     # training and EDA notebook
├── requirements.txt                # Python dependencies (optimized for Render CPU)
├── render.yaml                     # 1-click Render blueprint specification
├── .gitignore                      # excludes large datasets and cache
└── README.md
```

---

## Running it locally

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/ahlawatansh/AeroNet.git
cd AeroNet
```

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the server:
```bash
python app.py
```
Or directly with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at `http://localhost:8000`.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the main surveillance dashboard UI |
| `POST` | `/api/detect` | Multi-part image/snapshot detection with YOLOv8 localization & ResNet-18 classification |
| `POST` | `/classify` | Form-based image upload and HTML report rendering |
| `GET` | `/api/stats` | Returns real-time session telemetry, scan counts, and recent detections |

---

<div align="center">
<sub>Built by <a href="https://github.com/ahlawatansh">Ansh Ahlawat</a> · CSE undergrad, VIT Vellore</sub>
<br>
<sub><a href="https://www.linkedin.com/in/anshahlawat">LinkedIn</a> • <a href="https://github.com/ahlawatansh">GitHub</a></sub>
</div>
