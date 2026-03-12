"""
Step 2 — Face Landmarks using MediaPipe FaceMesh (478 points).
Builds protection masks for eyes, brows, lips, nostrils.
Uses mp.tasks.vision.FaceLandmarker API (mediapipe >= 0.10.14).
"""

import cv2
import numpy as np
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mediapipe"
_LANDMARKER_MODEL = _MODEL_DIR / "face_landmarker.task"

# Key landmark groups (MediaPipe FaceMesh 478-point indices)
_EYE_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133,
             173, 157, 158, 159, 160, 161, 246]
_EYE_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249,
              263, 466, 388, 387, 386, 385, 384, 398]
_BROW_LEFT = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_BROW_RIGHT = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321,
               375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318,
               324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
_NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5,
         4, 19, 94, 370, 462, 250, 309, 392, 289,
         305, 290, 328, 326, 2, 99, 240, 75, 60]


def extract_landmarks(img_rgb: np.ndarray):
    """
    Run MediaPipe FaceLandmarker and return 478 landmarks + protection mask.

    Returns
    -------
    landmarks : np.ndarray (478, 2) — (x, y) pixel coords or None
    protect_mask : np.ndarray (H, W) uint8 — 255 where features should be protected
    info : str
    """
    import mediapipe as mp
    h, w = img_rgb.shape[:2]

    print("[Step 2] Face Landmarks — MediaPipe FaceLandmarker")

    if not _LANDMARKER_MODEL.exists():
        print(f"  ERROR: Model file not found: {_LANDMARKER_MODEL}")
        return None, np.zeros((h, w), dtype=np.uint8), "Model file missing."

    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_LANDMARKER_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as landmarker:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        print("  No face landmarks detected.")
        return None, np.zeros((h, w), dtype=np.uint8), "No landmarks detected."

    face_lms = result.face_landmarks[0]
    pts = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in face_lms],
        dtype=np.int32,
    )  # shape (478, 2)

    # Build protection mask
    protect = np.zeros((h, w), dtype=np.uint8)
    dilation_px = max(3, int(min(h, w) * 0.008))
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )

    for group in [_EYE_LEFT, _EYE_RIGHT, _BROW_LEFT, _BROW_RIGHT,
                  _LIPS_OUTER, _LIPS_INNER, _NOSE]:
        hull = cv2.convexHull(pts[group])
        cv2.fillConvexPoly(protect, hull, 255)
    protect = cv2.dilate(protect, kern, iterations=1)

    # Debug: draw landmarks + mask
    debug = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for x, y in pts:
        cv2.circle(debug, (x, y), 1, (0, 255, 0), -1)
    cv2.imwrite(str(DEBUG_DIR / "step2_landmarks.jpg"), debug)

    protect_vis = cv2.merge([np.zeros_like(protect), np.zeros_like(protect), protect])
    overlay = cv2.addWeighted(debug, 0.7, protect_vis, 0.3, 0)
    cv2.imwrite(str(DEBUG_DIR / "step2_protect_mask.jpg"), overlay)
    cv2.imwrite(str(DEBUG_DIR / "step2_protect_mask.png"), protect)

    info = f"Landmarks: {len(pts)} points detected.\nProtection mask covers eyes, brows, lips, nose."
    print(f"[Step 2] Done — {len(pts)} landmarks, protect mask {np.count_nonzero(protect)} px")
    return pts, protect, info
