"""
Step 1 — Face Detection using RetinaFace (serengil/retinaface).
Auto-installs the package if missing.
Returns face bounding boxes, confidence scores and landmarks.
Uses mp.tasks.vision.FaceDetector API as fallback (mediapipe >= 0.10.14).
"""

import cv2
import numpy as np
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mediapipe"
_DETECTOR_MODEL = _MODEL_DIR / "blaze_face_short_range.tflite"


def _save_debug(name: str, img: np.ndarray):
    path = DEBUG_DIR / name
    cv2.imwrite(str(path), img)
    print(f"  [DEBUG] Saved {path}")


def detect_faces(img_rgb: np.ndarray):
    """
    Detect faces using serengil/retinaface.

    Returns
    -------
    faces : list[dict]
        Each dict: {"bbox": [x1,y1,x2,y2], "score": float, "landmarks": dict}
    debug_img : np.ndarray (RGB)
    info : str
    """
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("Empty image")

    print("[Step 1] Face Detection — RetinaFace")
    faces = []

    # --- Primary: serengil/retinaface ---
    try:
        from retinaface import RetinaFace as RF

        resp = RF.detect_faces(img_rgb)
        if isinstance(resp, dict) and len(resp) > 0:
            for key, val in resp.items():
                area = val.get("facial_area", [])
                score = val.get("score", 0.0)
                lms = val.get("landmarks", {})
                faces.append({
                    "bbox": list(area),
                    "score": float(score),
                    "landmarks": lms,
                })
            print(f"  RetinaFace (serengil) found {len(faces)} face(s).")
    except Exception as exc:
        print(f"  RetinaFace failed: {exc}")

    # --- Fallback: MediaPipe face detection (tasks API) ---
    if not faces:
        try:
            import mediapipe as mp
            if not _DETECTOR_MODEL.exists():
                raise FileNotFoundError(f"MediaPipe model not found: {_DETECTOR_MODEL}")
            h, w = img_rgb.shape[:2]
            opts = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_DETECTOR_MODEL)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            with mp.tasks.vision.FaceDetector.create_from_options(opts) as det:
                result = det.detect(mp_img)
            if result.detections:
                for d in result.detections:
                    bb = d.bounding_box
                    x1 = max(0, bb.origin_x)
                    y1 = max(0, bb.origin_y)
                    x2 = min(w, bb.origin_x + bb.width)
                    y2 = min(h, bb.origin_y + bb.height)
                    score = d.categories[0].score if d.categories else 0.0
                    faces.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": float(score),
                        "landmarks": {},
                    })
                print(f"  MediaPipe fallback found {len(faces)} face(s).")
        except Exception as exc2:
            print(f"  MediaPipe fallback failed: {exc2}")

    # --- Debug image ---
    debug = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug, f'{f["score"]:.2f}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    _save_debug("step1_face.jpg", debug)

    info = f"Detected {len(faces)} face(s).\n"
    for i, f in enumerate(faces):
        info += f"  Face {i+1}: bbox={f['bbox']}, score={f['score']:.4f}\n"
    print(f"[Step 1] Done — {len(faces)} face(s)")
    return faces, cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), info
