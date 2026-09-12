from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from predict import DEFAULT_MODEL_PATH, predict_image, prepare_rgb
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parent
try:
    from ultralytics import YOLO
    yolo_weight_path = ROOT / "yolov8n.pt"
    YOLO_MODEL = YOLO(str(yolo_weight_path) if yolo_weight_path.exists() else "yolov8n.pt")
except ImportError:
    YOLO_MODEL = None


def warmup_models() -> None:
    try:
        dummy = Image.new("RGB", (320, 320), (128, 128, 128))
        if YOLO_MODEL is not None:
            YOLO_MODEL(dummy, imgsz=320, verbose=False)
        predict_image(dummy)
    except Exception:
        pass


warmup_models()

templates = Jinja2Templates(directory=str(ROOT / "templates"))


def render_template(name: str, context: dict, status_code: int = 200) -> HTMLResponse:
    request = context.get("request")
    return templates.TemplateResponse(
        request=request,
        name=name,
        context=context,
        status_code=status_code,
    )


app = FastAPI(title="AeroNet - Drone & Bird Defense Zone Surveillance")
app.mount("/samples", StaticFiles(directory=str(ROOT / "samples")), name="samples")
DASHBOARD_STATE = {
    "scans": 0,
    "birds": 0,
    "drones": 0,
    "accuracy_sum": 0.0,
    "recent": [],
}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_dashboard_stats() -> dict:
    return {
        "stats": {
            "scans": DASHBOARD_STATE["scans"],
            "birds": DASHBOARD_STATE["birds"],
            "drones": DASHBOARD_STATE["drones"],
            "avg_accuracy": (
                DASHBOARD_STATE["accuracy_sum"] / DASHBOARD_STATE["scans"] * 100
                if DASHBOARD_STATE["scans"]
                else 0.0
            ),
        },
        "recent_detections": DASHBOARD_STATE["recent"][:5],
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    context = load_dashboard_stats()
    return render_template(
        "index.html",
        {
            "request": request,
            "result": None,
            "error": None,
            "model_ready": DEFAULT_MODEL_PATH.exists(),
            "status_message": "Ready to detect." if DEFAULT_MODEL_PATH.exists() else None,
            "uploaded_image": None,
            **context,
        },
    )


@app.post("/", response_class=HTMLResponse)
async def classify(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    context = load_dashboard_stats()
    if not DEFAULT_MODEL_PATH.exists():
        return render_template(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": f"Model not found at {DEFAULT_MODEL_PATH}. Run train.py first.",
                "model_ready": False,
                "status_message": None,
                "uploaded_image": None,
                **context,
            },
            status_code=400,
        )

    try:
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return render_template(
                "index.html",
                {
                    "request": request,
                    "result": None,
                    "error": None,
                    "model_ready": True,
                    "status_message": "Invalid file type.",
                    "uploaded_image": None,
                    **context,
                },
                status_code=400,
            )

        raw_bytes = await image.read()
        uploaded = Image.open(BytesIO(raw_bytes))
        uploaded = prepare_rgb(uploaded)
        result = predict_image(uploaded)
        DASHBOARD_STATE["scans"] += 1
        DASHBOARD_STATE["accuracy_sum"] += float(result["confidence"])
        if result["label"] == "bird":
            DASHBOARD_STATE["birds"] += 1
        elif result["label"] == "drone":
            DASHBOARD_STATE["drones"] += 1
        DASHBOARD_STATE["recent"].insert(
            0,
            {
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "file_name": image.filename or "uploaded image",
            },
        )
        DASHBOARD_STATE["recent"] = DASHBOARD_STATE["recent"][:8]
        context = load_dashboard_stats()
        encoded_image = base64.b64encode(raw_bytes).decode("ascii")
        uploaded_image = f"data:image/{suffix.lstrip('.')};base64,{encoded_image}"
    except Exception as exc:
        return render_template(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": f"Could not process the image: {exc}",
                "model_ready": True,
                "status_message": "Invalid file type.",
                "uploaded_image": None,
                **context,
            },
            status_code=400,
        )

    status_text = f"Detected - it's a {result['label']}."

    return render_template(
        "index.html",
        {
            "request": request,
            "result": result,
            "error": None,
            "model_ready": True,
            "status_message": status_text,
            "uploaded_image": uploaded_image,
            **context,
        },
    )


@app.post("/classify", response_class=HTMLResponse)
async def classify_route(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    return await classify(request=request, image=image)


@app.get("/api/stats")
async def api_stats():
    return load_dashboard_stats()


@app.post("/api/detect")
async def api_detect(image: UploadFile = File(...)):
    if not DEFAULT_MODEL_PATH.exists():
        return JSONResponse({"error": f"Model not found at {DEFAULT_MODEL_PATH}. Run train.py first."}, status_code=400)

    try:
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return JSONResponse({"error": "Invalid file type."}, status_code=400)

        raw_bytes = await image.read()
        uploaded = Image.open(BytesIO(raw_bytes))
        uploaded = prepare_rgb(uploaded)

        has_box = False
        box_coords = None

        if YOLO_MODEL is not None:
            results = YOLO_MODEL(uploaded, imgsz=640, verbose=False, conf=0.15)
            boxes = results[0].boxes
            if len(boxes) > 0:
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                candidates = [i for i, c in enumerate(classes) if c in (14, 4, 33)]
                if not candidates:
                    candidates = list(range(len(classes)))
                best_idx = max(candidates, key=lambda i: (1 if classes[i] in (14, 4) else 0, confidences[i]))
                box = boxes.xyxy[best_idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                has_box = True
                box_coords = [x1, y1, x2, y2]

                bw, bh = x2 - x1, y2 - y1
                pad_x = max(20, int(bw * 0.35))
                pad_y = max(20, int(bh * 0.35))
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(uploaded.width, x2 + pad_x)
                cy2 = min(uploaded.height, y2 + pad_y)

                if cx2 > cx1 + 4 and cy2 > cy1 + 4:
                    cropped = uploaded.crop((cx1, cy1, cx2, cy2))
                    result = predict_image(cropped)
                else:
                    result = predict_image(uploaded)

                cv_img = cv2.cvtColor(np.array(uploaded), cv2.COLOR_RGB2BGR)
                color = (0, 0, 255) if result["label"] == "drone" else (0, 255, 0)
                thickness = max(2, int(min(cv_img.shape[0], cv_img.shape[1]) * 0.005))
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, thickness)

                label_text = f"{result['label'].upper()} ({result['confidence']*100:.1f}%)"
                font_scale = max(0.5, min(cv_img.shape[0], cv_img.shape[1]) * 0.001)
                cv2.putText(cv_img, label_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, int(thickness / 2)))
                uploaded = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        if not has_box:
            result = predict_image(uploaded)

        DASHBOARD_STATE["scans"] += 1
        DASHBOARD_STATE["accuracy_sum"] += float(result["confidence"])
        if result["label"] == "bird":
            DASHBOARD_STATE["birds"] += 1
        elif result["label"] == "drone":
            DASHBOARD_STATE["drones"] += 1

        DASHBOARD_STATE["recent"].insert(
            0,
            {
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "file_name": image.filename or "uploaded image",
            },
        )
        DASHBOARD_STATE["recent"] = DASHBOARD_STATE["recent"][:8]

        img_byte_arr = BytesIO()
        uploaded.save(img_byte_arr, format="JPEG")
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode("ascii")
        uploaded_image_data = f"data:image/jpeg;base64,{encoded_image}"

        return {
            "result": result,
            "has_box": has_box,
            "bbox": box_coords,
            "frame_dims": [uploaded.width, uploaded.height],
            "uploaded_image": uploaded_image_data,
            "stats": load_dashboard_stats(),
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Could not process the image: {exc}"}, status_code=400)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Could not process the image: {exc}"}, status_code=400)


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


