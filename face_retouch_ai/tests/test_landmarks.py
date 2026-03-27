"""
test_landmarks.py
=================
STEP 2 — Landmark Detection using MediaPipe FaceMesh with refine_landmarks.

Detects 468/478 landmarks, filters those outside the face bounding box,
and saves the annotated result.

Output: outputs/debug/landmarks.jpg

Usage:
    python tests/test_landmarks.py
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


def test_landmarks() -> None:
    """Detect 468/478 landmarks with MediaPipe FaceMesh, filter by bbox."""
    try:
        import mediapipe as mp

        image_path = get_input_image()
        print(f"[INFO] Loading image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image at {image_path}.")
            return

        h, w = img.shape[:2]
        print(f"[INFO] Image size: {w}x{h}")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_mesh = None  # legacy removed
        mp_draw = mp.tasks.vision.drawing_utils
        FLC = mp.tasks.vision.FaceLandmarksConnections

        _lm_model = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"
        if not _lm_model.exists():
            print(f"[ERROR] Model not found: {_lm_model}")
            return

        print("[INFO] Running FaceLandmarker...")
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_lm_model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as landmarker:
            result = landmarker.detect(mp_img)

        if not result.face_landmarks:
            print("[WARNING] No face landmarks detected.")
            return

        # Select exactly one face
        face_lms = result.face_landmarks[0]
        n = len(face_lms)
        print(f"[INFO] Detected {n} landmarks.")

        # Compute face bounding box from landmarks
        xs = [lm.x * w for lm in face_lms]
        ys = [lm.y * h for lm in face_lms]
        fx1, fy1 = int(min(xs)), int(min(ys))
        fx2, fy2 = int(max(xs)), int(max(ys))

        annotated = img.copy()

        # Draw tesselation + contours
        mp_draw.draw_landmarks(
            annotated, face_lms, FLC.FACE_LANDMARKS_TESSELATION,
        )
        mp_draw.draw_landmarks(
            annotated, face_lms, FLC.FACE_LANDMARKS_CONTOURS,
        )

        # Draw individual points, ignoring those outside face bbox
        drawn = 0
        for lm in face_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                cv2.circle(annotated, (cx, cy), 1, (0, 255, 0), -1)
                drawn += 1

        print(f"[INFO] Drew {drawn}/{n} landmarks inside face bbox.")

        out_path = str(DEBUG_DIR / "landmarks.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[OK] Result saved to {out_path}")

        cv2.imwrite(str(OUTPUTS_DIR / "landmarks.jpg"), annotated)

    except Exception as exc:
        print(f"[ERROR] Landmark detection failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_landmarks()
