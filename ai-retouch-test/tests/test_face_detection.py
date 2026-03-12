"""
test_face_detection.py
======================
STEP 1 — Face Detection with fallback chain.

Primary:  RetinaFace (InsightFace buffalo_l)
Fallback: MediaPipe FaceMesh bounding-box extraction

Output: outputs/debug/face_detection.jpg

Usage:
    python tests/test_face_detection.py
"""

import sys
import glob
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEBUG_DIR = OUTPUTS_DIR / "debug"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def get_input_image() -> str:
    """Return path to the first image found in inputs/."""
    try:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            matches = glob.glob(str(INPUTS_DIR / ext))
            if matches:
                return matches[0]
        print("[ERROR] No image found in inputs/.")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def _detect_retinaface(img: np.ndarray):
    """Primary detector: RetinaFace via InsightFace."""
    try:
        from insightface.app import FaceAnalysis

        print("[INFO] Trying RetinaFace (InsightFace buffalo_l)...")
        try:
            app = FaceAnalysis(
                name="buffalo_l", root=str(MODELS_DIR),
                providers=["CPUExecutionProvider"],
            )
        except TypeError:
            app = FaceAnalysis(name="buffalo_l", root=str(MODELS_DIR))
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(img)
        if faces:
            return [(f.bbox.astype(int).tolist(), float(f.det_score)) for f in faces]
        print("[WARNING] RetinaFace detected 0 faces.")
        return None
    except Exception as exc:
        print(f"[WARNING] RetinaFace failed: {exc}")
        return None


def _detect_mediapipe(img_rgb: np.ndarray):
    """Fallback detector: MediaPipe FaceDetector (tasks API)."""
    try:
        import mediapipe as mp

        print("[INFO] Fallback: MediaPipe FaceDetector...")
        h, w = img_rgb.shape[:2]
        _det_model = PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite"
        if not _det_model.exists():
            print(f"[WARNING] Model not found: {_det_model}")
            return None
        opts = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_det_model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        with mp.tasks.vision.FaceDetector.create_from_options(opts) as det:
            result = det.detect(mp_img)
        if result.detections:
            faces = []
            for d in result.detections:
                bb = d.bounding_box
                x1 = max(0, bb.origin_x)
                y1 = max(0, bb.origin_y)
                x2 = min(w, bb.origin_x + bb.width)
                y2 = min(h, bb.origin_y + bb.height)
                score = d.categories[0].score if d.categories else 0.0
                faces.append(([x1, y1, x2, y2], float(score)))
            return faces
        print("[WARNING] MediaPipe detected 0 faces.")
        return None
    except Exception as exc:
        print(f"[WARNING] MediaPipe fallback failed: {exc}")
        return None


def test_face_detection() -> None:
    """Detect faces with RetinaFace → MediaPipe fallback, save annotated image."""
    try:
        image_path = get_input_image()
        print(f"[INFO] Loading image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image at {image_path}.")
            return
        print(f"[INFO] Image shape: {img.shape}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detection chain: RetinaFace → MediaPipe
        detections = _detect_retinaface(img)
        if not detections:
            detections = _detect_mediapipe(img_rgb)

        if not detections:
            print("[WARNING] No faces detected by any method.")
            return

        print(f"[INFO] Detected {len(detections)} face(s).")

        # Draw bounding boxes
        annotated = img.copy()
        for idx, (bbox, score) in enumerate(detections):
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
            label = f"Face {idx+1}: {score:.2f}"
            cv2.putText(annotated, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"       Face {idx+1}  bbox={bbox}  score={score:.4f}")

        # Save to debug directory
        out_path = str(DEBUG_DIR / "face_detection.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[OK] Result saved to {out_path}")

        # Also save to outputs/ for backward compatibility
        cv2.imwrite(str(OUTPUTS_DIR / "face_detection.jpg"), annotated)

    except Exception as exc:
        print(f"[ERROR] Face detection failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_face_detection()
