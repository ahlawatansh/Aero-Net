<div align="center">

# AeroNet — Real-Time Drone & Bird Defense Zone Surveillance

**An embedded edge-AI surveillance system combining YOLOv8 target localization, ResNet-18 visual feature extraction, and real-time classification to secure sensitive airspace.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-00FFFF.svg?logo=yolo)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg?logo=render)](https://render.com/)

###  Live: https://aeronet-ng9t.onrender.com/

</div>

---

## What it is

I built AeroNet to distinguish between low-flying unmanned aerial vehicles (drones) and natural birds in real time — one of the hardest challenges in perimeter security and airspace monitoring. Traditional radar struggles with small radar cross-sections at low altitudes, while naive computer vision models either choke on latency or misclassify birds as aerial threats.

AeroNet solves this with an edge-deployable, dual-stage vision architecture. It captures incoming aerial imagery, localizes candidate airborne objects in real time with YOLOv8 at ~15ms inference latency, extracts deep 512-dimensional visual embeddings using a ResNet-18 backbone, and classifies targets with a calibrated classifier — achieving 98.66% validation accuracy. When an unauthorized drone enters the sector, the system highlights the target reticle in red and updates real-time telemetry metrics.

---

## Features

**Dual-Stage Vision Pipeline** — decoupled localization and classification. YOLOv8 isolates regions of interest at 30+ FPS, while ResNet-18 deep feature extraction provides industrial-grade classification accuracy.

**Instant Threat Identification & Localization** — rapid upload analysis with automatic bounding box localization, overlaid classification badges, and probability breakdowns.

**Interactive Sample Dataset Testing** — one-click instant evaluation cards for preloaded drone, bird, and neutral background samples directly from the dashboard.

**High-Precision Classification** — trained on thousands of aerial drone and bird imagery samples, achieving 98.66% validation accuracy with sub-percent false positive rates.

**Session Telemetry** — tracks live scan count, threat detection ratios, running confidence averages, and real-time audit logs without persistent database clutter.

---

## How the Detection Algorithm Works (From Scratch to Advanced)

To understand how AeroNet tells apart a drone from a bird — even when they appear as tiny distant specks in the sky — let's break down the system from basic intuition up to the exact mathematical machinery.

```text
[ Input Image ]
       │
       ▼
[ Stage 1: YOLOv8 Target Spotter ] ────► Locates airborne signature coordinates (x, y, w, h)
       │
       ▼ (Crops target with adaptive context margin)
[ Stage 2: ResNet-18 Feature Microscope ] ────► Computes hierarchical convolutions & 512-dim embedding
       │
       ▼
[ Stage 3: Linear Decision Head ] ────► Computes dot product: z = (W · x) + b
       │
       ▼
[ Sigmoid Activation: 1 / (1 + e^-z) ] ────► Final calibrated probability (e.g. 99.8% Drone vs 0.2% Bird)
```

### 1. The Core Challenge: Why is this hard?
At a distance, both a bird and a drone look like tiny dark blobs against clouds or blue sky.
- A human eye struggles because both can hover, glide, or dart unpredictably.
- Traditional radar misses small commercial drones because plastic and carbon-fiber blades have a near-zero radar cross-section (RCS).
- Standard machine learning models trained on full images easily get distracted by clouds, trees, sun glare, or horizon lines.

AeroNet resolves this by splitting the problem into two specialized neural networks: **The Spotter** and **The Microscope**.

---

### 2. Stage 1: Target Localization (YOLOv8 — "The Spotter")
Before classifying what something is, the system first needs to know *where* it is.
1. **Grid Partitioning**: YOLOv8 divides the incoming image into a grid of cells. Each cell predicts bounding box anchors and an "objectness" probability score.
2. **Airspace Filtering**: It scans for anomalies that contrast against the background sky.
3. **Adaptive Context Padding**:
   - If you crop a tiny 15×15 pixel object too tightly, wingtips, rotors, and aerodynamic profiles get sliced off.
   - When scaled up, a tight crop turns into a blurred square blob.
   - AeroNet applies an adaptive context margin ($+35\%$ boundary expansion) around the detected box. This preserves the full silhouette and natural wing sweep against the ambient sky before passing it to the classifier.

---

### 3. Stage 2: Visual Structure Analysis (ResNet-18 — "The Microscope")
Does the algorithm check manual dot coordinates? **No.** Instead, it uses **Deep Convolutional Neural Networks (CNNs)** to extract hierarchical geometric features automatically:

#### A. Pixels as Numerical Matrices
Every image is treated as a grid of numbers from `0` to `255` across three color channels (Red, Green, Blue).

#### B. Hierarchical Feature Extraction
ResNet-18 slides small mathematical filters ($3 \times 3$ convolution kernels) across the pixels to calculate local gradient changes:
- **Shallow Layers (Lines & Gradients)**: Detect raw edge orientations. Distinguishes rigid straight lines and sharp $90^\circ$ angles from soft, curved feathered edges.
- **Middle Layers (Sub-Structures & Configurations)**: Combines lines into identifiable component shapes:
  - **Drone Signatures**: High geometric symmetry, perpendicular motor booms, cross ($X$ or $+$) airframe arms, central rectangular electronics pod, circular propeller rotor discs.
  - **Bird Signatures**: Organic curvature, tapered wing sweeps, pointed head/beak profiles, trailing tail feathers, non-perpendicular joints.
- **Deep Layers (Residual Skip Connections)**:
  In traditional deep networks, subtle shape signals degrade as layers get deeper ("vanishing gradient"). ResNet introduces shortcut connections:
  $$\text{Output} = \mathcal{F}(\vec{x}) + \vec{x}$$
  This skip-connection allows the network to preserve fine-grained structural details alongside high-level abstract shapes.

#### C. The 512-Dimensional "Digital Fingerprint" ($\vec{x}$)
At the final pooling layer, ResNet condenses the visual properties of the target into a vector of **512 real numbers**:
$$\vec{x} = [x_1, x_2, x_3, \dots, x_{512}]$$
Each number represents an abstract geometric attribute learned during training (e.g., $x_{42}$ correlates with carbon boom linearity, $x_{108}$ correlates with wing sweep curvature).

---

### 4. Stage 3: The Mathematical Decision (The Linear Classifier)
During training on thousands of verified aerial imagery samples, the model learned a set of **512 optimal weights** ($\vec{w}$) and a scalar bias ($b$):
- Features typical of drones receive **positive weights** ($+w$).
- Features typical of birds receive **negative weights** ($-w$).

When evaluating an unknown target, the model computes the **dot product** (multiplying each visual feature by its learned importance and summing them up):

$$z = (\vec{w} \cdot \vec{x}) + b = \sum_{i=1}^{512} (w_i \cdot x_i) + b$$

Geometric interpretation: Think of a 512-dimensional hyperspace. The vector $\vec{w}$ and bias $b$ define a flat dividing boundary (a hyperplane). Drones cluster on one side of this boundary; birds cluster on the other.

---

### 5. Stage 4: Probability Calibration (The Sigmoid Function)
The raw score $z$ can be any positive or negative number. To turn it into a calibrated probability percentage, the system passes $z$ through the **Sigmoid activation function**:

$$P(\text{Drone}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Case 1: Heavy Drone Characteristics**  
  $\implies z$ is positive and large (e.g., $z = +8.2$)  
  $\implies P(\text{Drone}) = \frac{1}{1 + e^{-8.2}} \approx 0.9997$ (**99.97% Drone**, **0.03% Bird**)
- **Case 2: Heavy Bird Characteristics**  
  $\implies z$ is negative and large (e.g., $z = -7.6$)  
  $\implies P(\text{Drone}) = \frac{1}{1 + e^{7.6}} \approx 0.0005$ (**0.05% Drone**, **99.95% Bird**)

---

### 6. Why it Works on Distant & Small Targets
Even when a target is far away and occupies only 1% to 5% of the frame:
1. **Aspect Ratio & Contour Rigidity**: The ratio of fuselage width to boom length in a drone remains rigid and invariant under distance, whereas bird aspect ratios follow aerodynamic wing-to-body ratios.
2. **Frequency Spectra**: Drone airframes present high-frequency linear edges against low-frequency atmospheric backgrounds.
3. **Contrast Preservation**: The adaptive context window preserves the object-to-sky boundary, preventing interpolation artifacts from corrupting the classifier.

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
| Frontend | HTML5, Vanilla ES6+, Modern CSS3 Glassmorphism |
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
│   └── index.html                  # single-page cyberpunk surveillance UI & analysis
├── samples/
│   ├── sample_1.jpg                # distant airspace surveillance target 01
│   ├── sample_2.jpg                # distant airspace surveillance target 02
│   ├── sample_3.jpg                # distant airspace surveillance target 03
│   ├── sample_4.jpg                # distant airspace surveillance target 04
│   ├── sample_5.jpg                # distant airspace surveillance target 05
│   └── sample_6.jpg                # distant airspace surveillance target 06
├── app.py                          # FastAPI application and API endpoints
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
| `POST` | `/api/detect` | Multi-part image upload detection with YOLOv8 localization & ResNet-18 classification |
| `POST` | `/classify` | Form-based image upload and HTML report rendering |
| `GET` | `/api/stats` | Returns real-time session telemetry, scan counts, and recent detections |

---

<div align="center">
<sub>Built by <a href="https://github.com/ahlawatansh">Ansh Ahlawat</a> · CSE undergrad, VIT Vellore</sub>
<br>
<sub><a href="https://www.linkedin.com/in/anshahlawat">LinkedIn</a> • <a href="https://github.com/ahlawatansh">GitHub</a></sub>
</div>
