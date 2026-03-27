"""
app.py — AI Portrait Retouch Pipeline (v3)
============================================
Blemish pipeline tuned toward Evoto-style portraits: even tone, strong redness cleanup,
controlled smoothing (landmarks protect eyes/brows/lips/nose).

Pipeline:
  1. Face Detection     (RetinaFace → MediaPipe fallback)
  2. Landmarks          (MediaPipe Face Landmarker .task)
  3. Face Parsing       (BiSeNet)
  4–6. Blemish + expand + inpaint (LaMa / residual pass)
  7. Redness correction (LAB A + global pass)
  8. Guided-filter smooth (texture weight slider)
  9. Tone unify         (optional checkbox — Evoto-like even chroma)
  10. Texture restore   (clamped high-pass)
  11. GFPGAN            (low–moderate blend)
"""

import sys
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from pipelines.face_restore import restore_face as restore_face_backend

try:
    from pipelines.skin_tone_harmonizer import harmonize_skin_tone_model
except Exception:
    harmonize_skin_tone_model = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEBUG_DIR = OUTPUTS_DIR / "debug"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe 0.10+ Windows wheels ship Tasks only (no mp.solutions / mediapipe.framework).
_LANDMARKER_CANDIDATES = (MODELS_DIR / "mediapipe" / "face_landmarker.task",)
_LM_DL_HINT = (
    "Place face_landmarker.task in models/mediapipe/ "
    "(download from the MediaPipe Face Landmarker model page)."
)

# 478-point mesh groups — aligned with face_retouch_ai/pipelines/landmarks.py
_LM_EYE_L = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_LM_EYE_R = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
_LM_BROW_L = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
_LM_BROW_R = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
_LM_LIPS_OUT = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)
_LM_LIPS_IN = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)
_LM_NOSE = (
    1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 370, 462, 250, 309, 392, 289,
    305, 290, 328, 326, 2, 99, 240, 75, 60,
)


def _resolve_face_landmarker_path() -> Path | None:
    for p in _LANDMARKER_CANDIDATES:
        if p.is_file():
            return p
    return None


def _mediapipe_landmark_points(img_rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
    """478-point landmarks via MediaPipe Tasks FaceLandmarker. Returns (pts (N,2) int32, info)."""
    path = _resolve_face_landmarker_path()
    if path is None:
        return None, f"Model missing. {_LM_DL_HINT}"

    import mediapipe as mp

    h, w = img_rgb.shape[:2]
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as landmarker:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None, "No landmarks detected."

    face_lms = result.face_landmarks[0]
    pts = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in face_lms],
        dtype=np.int32,
    )
    n = pts.shape[0]
    if n < 200:
        return None, f"Too few landmarks ({n})."
    return pts, f"Landmarks OK ({n} points)."


def _build_feature_protect_mask_from_pts(pts: np.ndarray, h: int, w: int) -> np.ndarray:
    protect = np.zeros((h, w), dtype=np.uint8)
    dilation_px = max(3, int(min(h, w) * 0.008))
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    groups = (
        _LM_EYE_L,
        _LM_EYE_R,
        _LM_BROW_L,
        _LM_BROW_R,
        _LM_LIPS_OUT,
        _LM_LIPS_IN,
        _LM_NOSE,
    )
    max_i = pts.shape[0] - 1
    for group in groups:
        idx = [i for i in group if i <= max_i]
        if len(idx) < 3:
            continue
        hull = cv2.convexHull(pts[idx])
        cv2.fillConvexPoly(protect, hull, 255)
    return cv2.dilate(protect, kern, iterations=1)


# ---------------------------------------------------------------------------
# Default configuration — easy to tune
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Sliders below match "Evoto Target" — polished, even tone (lower texture_w => more smooth).
    "smooth_strength": 14,
    "texture_strength": 0.72,
    "sharpen_strength": 0.03,
    "blemish_ai_threshold": 0.21,
    "blemish_threshold": 138,   # A-channel floor (redness gate)
    "saturation_threshold": 45, # HSV saturation threshold
    "texture_threshold": 20,    # Normalized Laplacian bump threshold
    "blemish_score_floor": 0.98,  # lower => slightly more blemish candidates
    "blemish_max_area": 1200,   # keep larger acne clusters for strong removal
    "blemish_min_area": 10,     # remove connected components smaller than this
    "mask_dilate_size": 13,     # ellipse kernel size for mask expansion
    "max_expansion_ratio": 5.0,  # inpaint halo vs. healthy skin (Evoto removes aggressively)
    "inpaint_radius": 10,       # Telea inpainting radius
    "inpaint_backend": "lama",  # lama / telea (mat|zits|sd -> lama fallback)
    "guided_radius": 11,        # guided filter radius for skin smoothing
    "guided_eps": 0.042,        # guided filter epsilon
    "face_restore_blend": 0.10,  # GFPGAN subtle blend (slider overrides in full pipeline)
    "face_restore_method": "auto",  # auto / gfpgan / codeformer
    "codeformer_fidelity": 0.70,
    "redness_strength": 0.76,   # LAB A-channel neutralization
    "redness_gate_sigma": 0.55,  # lower => correct broader redness (Evoto-like evenness)
    "micro_spot_a_boost": 5.0,
    "micro_spot_max_area": 70,
    "global_redness_strength": 0.32,
    "second_inpaint_radius": 4,
    "tile_micro_tile": 96,
    "tile_micro_overlap": 24,
    "post_redness_pass_strength": 0.11,
    "enable_tile_micro": False,
    "enable_tone_unify": True,  # Step 9 — key for uniform skin like Evoto
    "tone_unify_strength": 0.30,
    "tone_unify_radius": 13,
    "highlight_rolloff": 0.045,
    "use_learned_tone_harmonizer": True,
    "learned_tone_strength": 0.32,
}

PROFILE_PRESETS = {
    "Natural": {
        "smooth": 7,
        "texture": 0.92,
        "sharpen": 0.05,
        "blemish": 0.36,
        "restore": 0.02,
    },
    "Studio": {
        "smooth": 9,
        "texture": 0.90,
        "sharpen": 0.05,
        "blemish": 0.31,
        "restore": 0.05,
    },
    "Evoto Target": {
        "smooth": 14,
        "texture": 0.72,
        "sharpen": 0.03,
        "blemish": 0.21,
        "restore": 0.10,
    },
}


def _resolve_blemish_model_source() -> str:
    """Return active blemish model source for UI/logging."""
    segformer_paths = (
        MODELS_DIR / "blemish_seg" / "segformer_acne",
        MODELS_DIR / "blemish_seg" / "segformer_b0_acne",
    )
    unet_paths = (
        MODELS_DIR / "blemish_seg" / "unet_blemish.pth",
        MODELS_DIR / "blemish_seg" / "unet_blemish_best.pth",
    )
    for p in segformer_paths:
        if p.exists():
            return f"SegFormer ({p.name})"
    for p in unet_paths:
        if p.exists():
            return f"U-Net ({p.name})"
    return "Heuristic fallback (khong tim thay model AI)"


# ---------------------------------------------------------------------------
# Utility: save debug image
# ---------------------------------------------------------------------------
def _save_debug(name: str, img: np.ndarray) -> str:
    """Save an image to the debug directory. Returns the saved path."""
    try:
        path = str(DEBUG_DIR / name)
        cv2.imwrite(path, img)
        print(f"  [DEBUG] Saved {path}")
        return path
    except Exception as exc:
        print(f"  [DEBUG] Failed to save {name}: {exc}")
        return ""


def _compute_local_stats(channel: np.ndarray, ksize: int = 31):
    """Compute local mean/std using box filter."""
    ch_f = channel.astype(np.float32)
    local_mean = cv2.blur(ch_f, (ksize, ksize))
    local_sq = cv2.blur(ch_f * ch_f, (ksize, ksize))
    local_var = np.maximum(local_sq - local_mean * local_mean, 0)
    return local_mean, np.sqrt(local_var)


def _component_filter(mask: np.ndarray, min_area: int, max_area: int) -> tuple[np.ndarray, int, int]:
    """Keep only connected components in [min_area, max_area]."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    removed_small = 0
    removed_large = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            removed_small += 1
            continue
        if area > max_area:
            removed_large += 1
            continue
        out[labels == i] = 255
    return out, removed_small, removed_large


def _keep_small_components(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Keep only small connected components in [min_area, max_area]."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            out[labels == i] = 255
    return out


def _tile_micro_spot_mask(a_ch: np.ndarray, lap_norm: np.ndarray, skin_bin: np.ndarray,
                          tile: int, overlap: int, max_area: int) -> np.ndarray:
    """Detect micro spots per local tile to catch low-contrast tiny acne."""
    h, w = a_ch.shape
    stride = max(16, tile - overlap)
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(h, y + tile)
            x2 = min(w, x + tile)
            skin_patch = skin_bin[y:y2, x:x2]
            if np.count_nonzero(skin_patch) < 350:
                continue
            a_patch = a_ch[y:y2, x:x2]
            l_patch = lap_norm[y:y2, x:x2]
            skin_vals = a_patch[skin_patch == 1]
            if skin_vals.size < 50:
                continue
            a_thr = float(np.mean(skin_vals) + 1.4 * np.std(skin_vals))
            l_thr = float(np.percentile(l_patch[skin_patch == 1], 75))
            patch_raw = ((a_patch > a_thr) & (l_patch > l_thr) & (skin_patch == 1)).astype(np.uint8) * 255
            patch_raw = _keep_small_components(patch_raw, min_area=3, max_area=min(max_area, 35))
            out[y:y2, x:x2] = cv2.bitwise_or(out[y:y2, x:x2], patch_raw)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Face Detection — RetinaFace → MediaPipe fallback
# ═══════════════════════════════════════════════════════════════════════════
def step_face_detection(img_rgb: np.ndarray):
    """Detect faces with RetinaFace; fall back to MediaPipe FaceMesh."""
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, "Error: empty image."

        print("[Step 1] Face Detection — start")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        detections = None

        # --- Primary: RetinaFace via InsightFace ---
        try:
            from insightface.app import FaceAnalysis

            print("  Trying RetinaFace (InsightFace buffalo_l)...")

            # Try multiple det_sizes (smaller catches close-up faces)
            for det_sz in [(640, 640), (320, 320)]:
                try:
                    fa = FaceAnalysis(
                        name="buffalo_l",
                        root=str(MODELS_DIR),
                        providers=["CPUExecutionProvider"],
                    )
                except TypeError:
                    fa = FaceAnalysis(name="buffalo_l", root=str(MODELS_DIR))
                fa.prepare(ctx_id=0, det_size=det_sz)
                faces = fa.get(img_bgr)
                if faces:
                    detections = [
                        (f.bbox.astype(int).tolist(), float(f.det_score)) for f in faces
                    ]
                    print(f"  RetinaFace found {len(detections)} face(s) at det_size={det_sz}.")
                    break
            if not detections:
                print("  RetinaFace detected 0 faces at all sizes.")
        except Exception as exc:
            print(f"  RetinaFace failed: {exc}")

        # --- Fallback: MediaPipe FaceDetector (tasks API) ---
        if not detections:
            try:
                import mediapipe as mp

                print("  Fallback: MediaPipe FaceDetector...")
                h, w = img_rgb.shape[:2]
                _det_model = MODELS_DIR / "mediapipe" / "blaze_face_short_range.tflite"
                if not _det_model.exists():
                    raise FileNotFoundError(f"Model not found: {_det_model}")
                opts = mp.tasks.vision.FaceDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(_det_model)),
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
                        detections.append(([x1, y1, x2, y2], float(score)))
                    print(f"  MediaPipe fallback found {len(result.detections)} face(s).")
                else:
                    print("  MediaPipe fallback: no face detected.")
            except Exception as exc:
                print(f"  MediaPipe fallback failed: {exc}")

        if not detections:
            _save_debug("face_detection.jpg", img_bgr)
            return img_rgb, "No face detected by any method."

        # Draw bounding boxes
        annotated = img_bgr.copy()
        info_lines = []
        for i, (bbox, score) in enumerate(detections):
            cv2.rectangle(
                annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
            )
            label = f"Face {i+1}: {score:.2f}"
            cv2.putText(
                annotated, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            info_lines.append(f"Face {i+1}: bbox={bbox}, score={score:.4f}")

        _save_debug("face_detection.jpg", annotated)
        result_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        info = f"Detected {len(detections)} face(s).\n" + "\n".join(info_lines)
        print(f"[Step 1] Done — {len(detections)} face(s)")
        return result_rgb, info

    except Exception as exc:
        print(f"[Step 1] ERROR: {exc}")
        return img_rgb, f"Face detection error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Landmark Detection — MediaPipe FaceMesh 478
# ═══════════════════════════════════════════════════════════════════════════
def step_landmarks(img_rgb: np.ndarray):
    """Detect ~478 facial landmarks (MediaPipe Face Landmarker, Tasks API)."""
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, "Error: empty image."

        print("[Step 2] Landmark Detection — FaceLandmarker (tasks API)")
        pts, msg = _mediapipe_landmark_points(img_rgb)
        if pts is None:
            print(f"[Step 2] ERROR: {msg}")
            return img_rgb, f"Landmark detection error: {msg}"

        annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        fx1, fy1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        fx2, fy2 = int(pts[:, 0].max()), int(pts[:, 1].max())

        drawn = 0
        for cx, cy in pts:
            if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                cv2.circle(annotated, (int(cx), int(cy)), 1, (0, 255, 0), -1)
                drawn += 1

        _save_debug("landmarks.jpg", annotated)
        result_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        n = int(pts.shape[0])
        info = f"Detected {n} landmarks ({drawn} drawn). {msg}"
        print(f"[Step 2] Done — {n} landmarks")
        return result_rgb, info

    except Exception as exc:
        print(f"[Step 2] ERROR: {exc}")
        return img_rgb, f"Landmark detection error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Face Parsing — BiSeNet 512×512 + morphology refinement
# ═══════════════════════════════════════════════════════════════════════════
_bisenet_model = None


def _get_bisenet():
    """Lazy-load BiSeNet face parsing model."""
    global _bisenet_model
    if _bisenet_model is not None:
        return _bisenet_model

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models as tv_models

    class ConvBNReLU(nn.Module):
        def __init__(self, ci, co, ks=3, s=1, p=1):
            super().__init__()
            self.conv = nn.Conv2d(ci, co, ks, s, p, bias=False)
            self.bn = nn.BatchNorm2d(co)
        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class ARM(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.conv = ConvBNReLU(ci, co)
            self.conv_atten = nn.Conv2d(co, co, 1, bias=False)
            self.bn_atten = nn.BatchNorm2d(co)
        def forward(self, x):
            f = self.conv(x)
            a = torch.sigmoid(self.bn_atten(self.conv_atten(
                torch.mean(f, dim=[2, 3], keepdim=True))))
            return f * a

    class ContextPath(nn.Module):
        def __init__(self):
            super().__init__()
            r = tv_models.resnet18(weights=None)
            self.conv1, self.bn1, self.relu = r.conv1, r.bn1, r.relu
            self.maxpool = r.maxpool
            self.layer1 = r.layer1
            self.layer2 = r.layer2
            self.layer3 = r.layer3
            self.layer4 = r.layer4
            self.arm16 = ARM(256, 128)
            self.arm32 = ARM(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128)
            self.conv_head16 = ConvBNReLU(128, 128)
            self.conv_avg = ConvBNReLU(512, 128, ks=1, s=1, p=0)
        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            f8 = self.layer2(x)
            f16 = self.layer3(f8)
            f32 = self.layer4(f16)
            avg = self.conv_avg(torch.mean(f32, dim=[2, 3], keepdim=True))
            f32_up = self.conv_head32(
                self.arm32(f32) + F.interpolate(avg, size=f32.shape[2:], mode="nearest"))
            f32_up = F.interpolate(f32_up, size=f16.shape[2:], mode="nearest")
            f16_up = self.conv_head16(self.arm16(f16) + f32_up)
            f16_up = F.interpolate(f16_up, size=f8.shape[2:], mode="nearest")
            return f8, f16_up, f32_up

    class SpatialPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvBNReLU(3, 64, 7, 2, 3)
            self.conv2 = ConvBNReLU(64, 64, 3, 2, 1)
            self.conv3 = ConvBNReLU(64, 64, 3, 2, 1)
            self.conv_out = ConvBNReLU(64, 128)
        def forward(self, x):
            return self.conv_out(self.conv3(self.conv2(self.conv1(x))))

    class FFM(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.convblk = ConvBNReLU(ci, co, ks=1, s=1, p=0)
            self.conv1 = nn.Conv2d(co, co // 4, 1, bias=False)
            self.conv2 = nn.Conv2d(co // 4, co, 1, bias=False)
        def forward(self, fsp, fcp):
            f = self.convblk(torch.cat([fsp, fcp], 1))
            a = torch.sigmoid(self.conv2(F.relu(self.conv1(
                torch.mean(f, dim=[2, 3], keepdim=True)))))
            return f + f * a

    class Out(nn.Module):
        def __init__(self, ci, mi, nc):
            super().__init__()
            self.conv = ConvBNReLU(ci, mi)
            self.conv_out = nn.Conv2d(mi, nc, 1, bias=False)
        def forward(self, x):
            return self.conv_out(self.conv(x))

    class BiSeNet(nn.Module):
        def __init__(self, nc=19):
            super().__init__()
            self.cp = ContextPath()
            self.sp = SpatialPath()
            self.ffm = FFM(256, 256)
            self.conv_out = Out(256, 256, nc)
            self.conv_out16 = Out(128, 64, nc)
            self.conv_out32 = Out(128, 64, nc)
        def forward(self, x):
            H, W = x.shape[2:]
            f8, fcp8, _ = self.cp(x)
            fsp = self.sp(x)
            return F.interpolate(
                self.conv_out(self.ffm(fsp, fcp8)),
                (H, W), mode="bilinear", align_corners=True,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiSeNet(19).to(device).eval()
    ckpt = MODELS_DIR / "face_parsing" / "79999_iter.pth"
    if not ckpt.exists():
        ckpt = MODELS_DIR / "face_parsing_79999_iter.pth"
    if ckpt.exists():
        sd = torch.load(str(ckpt), map_location=device, weights_only=True)
        # Remap checkpoint keys: cp.resnet.X → cp.X
        remapped = {}
        for k, v in sd.items():
            new_key = k.replace("cp.resnet.", "cp.")
            remapped[new_key] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            print(f"  [BiSeNet] Missing keys: {len(missing)}")
        if unexpected:
            print(f"  [BiSeNet] Unexpected keys: {len(unexpected)}")
    _bisenet_model = (model, device)
    return _bisenet_model


# Face parsing label names
_LABEL_NAMES = {
    0: "background", 1: "skin", 2: "l_brow", 3: "r_brow",
    4: "l_eye", 5: "r_eye", 6: "eyeglass", 7: "l_ear", 8: "r_ear",
    9: "earring", 10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip",
    14: "neck", 15: "necklace", 16: "cloth", 17: "hair", 18: "hat",
}

# Labels to EXCLUDE from blemish detection AND smoothing
# eyes + eyebrows + lips + nostrils(nose) + hair
_EXCLUDE_LABELS = {2, 3, 4, 5, 10, 11, 12, 13, 17}


def step_face_parsing(img_rgb: np.ndarray):
    """BiSeNet parsing at 512x512 with morphological refinement."""
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, np.zeros(img_rgb.shape[:2], dtype=np.uint8), "Error: empty image."

        print("[Step 3] Face Parsing — start")
        import torch
        from torchvision import transforms

        model, device = _get_bisenet()
        h, w = img_rgb.shape[:2]

        # Preprocess: resize to 512x512
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        inp = transform(img_rgb).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            out = model(inp)
        parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

        # Colour segmentation mask
        np.random.seed(42)
        palette = np.random.randint(0, 255, (20, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]
        colour_mask = cv2.resize(palette[parsing], (w, h), interpolation=cv2.INTER_NEAREST)
        colour_mask_rgb = cv2.cvtColor(colour_mask, cv2.COLOR_BGR2RGB)

        # Extract skin-only mask (label 1)
        skin_raw = np.zeros_like(parsing, dtype=np.uint8)
        skin_raw[parsing == 1] = 255
        skin_raw = cv2.resize(skin_raw, (w, h), interpolation=cv2.INTER_NEAREST)

        # Morphological refinement to prevent mask leakage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_raw, cv2.MORPH_OPEN, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        _save_debug("skin_mask.png", skin_mask)
        _save_debug("face_parsing_mask.jpg", colour_mask)

        # Build exclusion mask (eyes, brows, lips, nose, hair)
        exclude = np.zeros_like(parsing, dtype=np.uint8)
        for lbl in _EXCLUDE_LABELS:
            exclude[parsing == lbl] = 255
        exclude = cv2.resize(exclude, (w, h), interpolation=cv2.INTER_NEAREST)
        _save_debug("no_smooth_mask.png", exclude)

        # Final skin mask = skin − exclusion (strict region)
        skin_mask = cv2.subtract(skin_mask, exclude)
        _save_debug("skin_mask_strict.png", skin_mask)

        # Stats
        unique, counts = np.unique(parsing, return_counts=True)
        info = "Detected regions:\n"
        for u, c in zip(unique, counts):
            pct = c / parsing.size * 100
            info += f"  {_LABEL_NAMES.get(u, f'class_{u}')}: {pct:.1f}%\n"
        info += f"\nSkin mask pixels: {np.count_nonzero(skin_mask)}"

        print(f"[Step 3] Done — skin pixels: {np.count_nonzero(skin_mask)}")
        return colour_mask_rgb, skin_mask, info

    except Exception as exc:
        h, w = img_rgb.shape[:2] if img_rgb is not None else (1, 1)
        print(f"[Step 3] ERROR: {exc}")
        return img_rgb, np.zeros((h, w), dtype=np.uint8), f"Face parsing error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Blemish Detection — LAB redness + Laplacian texture
# ═══════════════════════════════════════════════════════════════════════════
def step_blemish_detection(img_rgb: np.ndarray, skin_mask: np.ndarray = None,
                           threshold: float = None, var_threshold: float = None):
    """
        Combined blemish detection:
            1. AI segmentation (SegFormer/U-Net)
            2. LAB redness detection
            3. Laplacian texture detection
            4. Merge masks with OR
            4. Restrict to skin region and filter connected components
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return (np.zeros((1, 1), dtype=np.uint8),
                    img_rgb, "Error: empty image.")

        ai_threshold = threshold if threshold is not None else DEFAULT_CONFIG["blemish_ai_threshold"]
        a_floor = DEFAULT_CONFIG["blemish_threshold"]
        tex_thresh = DEFAULT_CONFIG["texture_threshold"]
        min_area = DEFAULT_CONFIG["blemish_min_area"]
        max_area = DEFAULT_CONFIG["blemish_max_area"]

        print("[Step 4] Blemish Detection — combine AI model + heuristic")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # ── Prepare skin mask ─────────────────────────────────────
        if skin_mask is not None:
            if len(skin_mask.shape) == 3:
                skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
            skin_mask_resized = cv2.resize(skin_mask, (w, h),
                                           interpolation=cv2.INTER_NEAREST)
        else:
            skin_mask_resized = np.full((h, w), 255, dtype=np.uint8)

        skin_bin = (skin_mask_resized > 127).astype(np.uint8)
        skin_count = np.count_nonzero(skin_bin)

        if skin_count == 0:
            print("[Step 4] No skin pixels — returning empty mask.")
            return (np.zeros((h, w), dtype=np.uint8), img_rgb,
                    "No skin pixels found.")

        ai_mask = np.zeros((h, w), dtype=np.uint8)
        ai_debug = None
        try:
            from pipelines.blemish_seg import detect_blemish_ai

            ai_mask, ai_debug = detect_blemish_ai(
                img_rgb,
                skin_mask=skin_mask_resized,
                threshold=ai_threshold,
                min_area=min_area,
                max_area=max_area,
                dilate_px=1,
            )
            print(
                f"  AI mask: regions={ai_debug['region_count']}, "
                f"prob=[{ai_debug['prob_min']:.3f}, {ai_debug['prob_max']:.3f}]"
            )
        except Exception as exc:
            print(f"  AI model unavailable: {exc}")

        # ── Condition 1: LAB A-channel relative redness ────────────
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        A_ch = lab[:, :, 1].astype(np.float32)
        L_ch = lab[:, :, 0].astype(np.float32)
        red_score = A_ch - 128.0
        a_mean, a_std = _compute_local_stats(A_ch, ksize=31)
        redness_rel = np.clip((A_ch - a_mean) / np.maximum(a_std, 2.0), 0, 5.0)
        abs_gate = (A_ch >= a_floor).astype(np.float32)

        skin_a = A_ch[skin_bin == 1]
        print(f"  A-channel on skin: mean={np.mean(skin_a):.1f}, "
              f"std={np.std(skin_a):.1f}, "
              f"floor pixels={int(np.sum(abs_gate * skin_bin.astype(np.float32)))}")

        # ── Condition 2: HSV saturation anomaly ────────────────────
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S_ch = hsv[:, :, 1].astype(np.float32)
        s_mean, s_std = _compute_local_stats(S_ch, ksize=31)
        sat_rel = np.clip((S_ch - s_mean) / np.maximum(s_std, 2.0), 0, 5.0)

        # ── Condition 3: Laplacian texture anomaly ────────────────
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F)).astype(np.float32)
        lap_blur = cv2.GaussianBlur(lap, (5, 5), 0)
        lap_norm = cv2.normalize(lap_blur, None, 0, 255, cv2.NORM_MINMAX)
        skin_lap = lap_norm[skin_bin == 1]
        lap_gate = float(np.percentile(skin_lap, 70)) if skin_lap.size else float(tex_thresh)
        texture_rel = np.clip((lap_norm - lap_gate) / 18.0, 0, 5.0)

        # ── Condition 4: local darkness anomaly ───────────────────
        l_mean, l_std = _compute_local_stats(L_ch, ksize=31)
        dark_rel = np.clip((l_mean - L_ch) / np.maximum(l_std, 2.0), 0, 5.0)

        print(f"  Laplacian: mean={np.mean(lap[skin_bin == 1]):.1f}, "
              f"max={np.max(lap[skin_bin == 1]):.0f}, "
              f"gate={lap_gate:.1f}, texture>0 pixels={np.count_nonzero((texture_rel > 0) & (skin_bin == 1))}")

        score = (
            0.46 * redness_rel
            + 0.20 * sat_rel
            + 0.24 * texture_rel
            + 0.10 * dark_rel
        )
        score = score * abs_gate * skin_bin.astype(np.float32)

        nonzero_scores = score[(score > 0.1) & (skin_bin == 1)]
        score_floor = DEFAULT_CONFIG["blemish_score_floor"]
        if len(nonzero_scores) > 50:
            p75 = np.percentile(nonzero_scores, 75)
            p95 = np.percentile(nonzero_scores, 95)
            score_thresh = max(p75 + 0.45 * (p95 - p75), score_floor)
        else:
            score_thresh = score_floor + 0.15

        heuristic_mask = ((score > score_thresh) & (skin_bin == 1)).astype(np.uint8) * 255

        # Micro red spots (tiny inflamed dots often missed by U-Net thresholding)
        micro_thresh = skin_mean = float(np.mean(skin_a)) if skin_a.size else 128.0
        micro_thresh += DEFAULT_CONFIG["micro_spot_a_boost"]
        micro_raw = ((A_ch > micro_thresh) & (lap_norm > (lap_gate - 2.0)) & (skin_bin == 1)).astype(np.uint8) * 255

        # Blob-based tiny acne detector (helps with red pinpoint spots)
        red_vis = np.clip((A_ch - skin_mean) * 8.0, 0, 255).astype(np.uint8)
        blob_mask = np.zeros_like(micro_raw)
        try:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByColor = True
            params.blobColor = 255
            params.filterByArea = True
            params.minArea = 4
            params.maxArea = float(DEFAULT_CONFIG["micro_spot_max_area"])
            params.filterByCircularity = True
            params.minCircularity = 0.2
            params.filterByConvexity = False
            params.filterByInertia = False
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(red_vis)
            for kp in keypoints:
                cx, cy = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= cy < h and 0 <= cx < w and skin_bin[cy, cx] == 1:
                    r = max(2, int(kp.size * 0.6))
                    cv2.circle(blob_mask, (cx, cy), r, 255, -1)
        except Exception:
            keypoints = []
        micro_mask = _keep_small_components(
            micro_raw,
            min_area=max(3, min_area // 2),
            max_area=DEFAULT_CONFIG["micro_spot_max_area"],
        )
        blob_mask = _keep_small_components(
            blob_mask,
            min_area=3,
            max_area=DEFAULT_CONFIG["micro_spot_max_area"],
        )
        if DEFAULT_CONFIG.get("enable_tile_micro", False):
            tile_mask = _tile_micro_spot_mask(
                A_ch,
                lap_norm,
                skin_bin,
                tile=DEFAULT_CONFIG["tile_micro_tile"],
                overlap=DEFAULT_CONFIG["tile_micro_overlap"],
                max_area=DEFAULT_CONFIG["micro_spot_max_area"],
            )
        else:
            tile_mask = np.zeros_like(micro_raw)
        micro_mask = cv2.bitwise_or(micro_mask, blob_mask)
        micro_mask = cv2.bitwise_or(micro_mask, tile_mask)
        heuristic_mask = cv2.bitwise_or(heuristic_mask, micro_mask)

        # Merge AI + heuristic conservatively:
        # - If AI exists, trust AI as primary.
        # - Only keep heuristic regions near AI detections (or tiny micro spots).
        if np.count_nonzero(ai_mask) > 0:
            ai_dilate = cv2.dilate(
                ai_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            heuristic_near_ai = cv2.bitwise_and(heuristic_mask, ai_dilate)
            micro_only = _keep_small_components(
                micro_mask,
                min_area=3,
                max_area=DEFAULT_CONFIG["micro_spot_max_area"],
            )
            mask = cv2.bitwise_or(ai_mask, heuristic_near_ai)
            mask = cv2.bitwise_or(mask, micro_only)
        else:
            mask = heuristic_mask

        # ── Morphological cleanup ─────────────────────────────────
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # ── Connected-component filtering ─────────────────────────
        mask, removed_small, removed_large = _component_filter(mask, min_area, max_area)
        removed_shape = 0
        print(f"  Removed {removed_small} tiny (< {min_area} px), "
              f"{removed_large} large (> {max_area} px), "
              f"{removed_shape} non-circular regions")

        # ── Statistics ────────────────────────────────────────────
        n_blemish = np.count_nonzero(mask)
        pct_skin = n_blemish / max(skin_count, 1) * 100
        pct_total = n_blemish / mask.size * 100

        print(f"[Step 4] Blemish pixels: {n_blemish} ({pct_skin:.2f}% of skin)")

        # ── Debug visualizations ──────────────────────────────────
        overlay = img_bgr.copy()
        coloured = np.zeros_like(img_bgr)
        coloured[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.7, coloured, 0.3, 0)

        red_vis = np.clip(red_score * 10, 0, 255).astype(np.uint8)
        _save_debug("blemish_red_score.jpg",
                    cv2.applyColorMap(red_vis, cv2.COLORMAP_JET))
        _save_debug("blemish_mask.png", mask)
        _save_debug("blemish_overlay.jpg", overlay)

        ai_region_count = ai_debug["region_count"] if ai_debug is not None else 0
        heuristic_pixels = int(np.count_nonzero(heuristic_mask))
        ai_pixels = int(np.count_nonzero(ai_mask))
        micro_pixels = int(np.count_nonzero(micro_mask))
        tile_pixels = int(np.count_nonzero(tile_mask))
        info = (f"Method: AI + heuristic\n"
            f"Blemish pixels: {n_blemish}\n"
                f"% of skin area: {pct_skin:.2f}%\n"
                f"% of total image: {pct_total:.2f}%\n"
            f"AI threshold={ai_threshold:.2f}, ai_regions={ai_region_count}\n"
            f"AI pixels={ai_pixels}, Heuristic pixels={heuristic_pixels}, Micro pixels={micro_pixels}, tile_pixels={tile_pixels}, blobs={len(keypoints)}\n"
            f"Score threshold={score_thresh:.2f}, A_floor={a_floor}, lap_gate={lap_gate:.1f}")
        print(f"[Step 4] Combined mask pixels: {n_blemish} ({pct_skin:.2f}% of skin)")
        return mask, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), info

    except Exception as exc:
        print(f"[Step 4] ERROR: {exc}")
        h, w = img_rgb.shape[:2] if img_rgb is not None else (1, 1)
        return np.zeros((h, w), dtype=np.uint8), img_rgb, f"Blemish detection error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4b: Smart Mask Expansion — controlled dilation + soft edge blur
# ═══════════════════════════════════════════════════════════════════════════
def step_expand_mask(blemish_mask: np.ndarray, skin_mask: np.ndarray = None,
                     max_radius: int = None):
    """
    Controlled mask expansion:
      1. Dilate with 5×5 ellipse kernel, 1 iteration
      2. GaussianBlur(ksize=9) for soft blending edges
      3. Re-threshold at 32 to get clean binary mask
      4. Constrain to skin region
    """
    try:
        if blemish_mask is None or blemish_mask.size == 0:
            return blemish_mask, "No mask to expand."

        ks = DEFAULT_CONFIG["mask_dilate_size"]
        max_ratio = DEFAULT_CONFIG["max_expansion_ratio"]
        print(f"[Step 4b] Mask Expansion — ellipse {ks}x{ks}, 2 iter, max_ratio={max_ratio}")
        h, w = blemish_mask.shape[:2]
        original_count = np.count_nonzero(blemish_mask)

        if len(blemish_mask.shape) == 3:
            blemish_mask = cv2.cvtColor(blemish_mask, cv2.COLOR_RGB2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        expanded = cv2.dilate(blemish_mask, kernel, iterations=2)

        # Step 2: blur edges for soft inpainting transition
        expanded_blur = cv2.GaussianBlur(expanded, (11, 11), 0)
        expanded = (expanded_blur > 32).astype(np.uint8) * 255

        # Step 3: constrain to skin region
        if skin_mask is not None:
            if len(skin_mask.shape) == 3:
                skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
            skin_resized = cv2.resize(skin_mask, (w, h),
                                       interpolation=cv2.INTER_NEAREST)
            expanded = cv2.bitwise_and(expanded, expanded,
                                        mask=(skin_resized > 127).astype(np.uint8) * 255)

        # Step 4: cap expansion ratio
        expanded_count = np.count_nonzero(expanded)
        if original_count > 0 and expanded_count / original_count > max_ratio:
            # Erode back until within budget
            erode_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            while np.count_nonzero(expanded) / max(original_count, 1) > max_ratio:
                expanded = cv2.erode(expanded, erode_kern, iterations=1)
            expanded_count = np.count_nonzero(expanded)
            print(f"  Capped expansion at {max_ratio}x")

        ratio = expanded_count / max(original_count, 1)
        _save_debug("blemish_mask_expanded.png", expanded)

        info = (f"Original mask: {original_count} px\n"
                f"Expanded mask: {expanded_count} px\n"
                f"Expansion ratio: {ratio:.1f}x")
        print(f"[Step 4b] Done — {original_count} → {expanded_count} px ({ratio:.1f}x)")
        return expanded, info

    except Exception as exc:
        print(f"[Step 4b] ERROR: {exc}")
        return blemish_mask, f"Mask expansion error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Smart Inpainting — expanded mask, stronger radius
# ═══════════════════════════════════════════════════════════════════════════
def _lama_inpaint(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """LaMa inpainting via simple-lama-inpainting package."""
    try:
        from simple_lama_inpainting import SimpleLama
        from PIL import Image

        lama = SimpleLama()
        result = np.array(lama(Image.fromarray(img_rgb), Image.fromarray(mask).convert("L")))
        if result.shape[:2] != img_rgb.shape[:2]:
            result = cv2.resize(result, (img_rgb.shape[1], img_rgb.shape[0]))
        return result
    except Exception as exc:
        print(f"  LaMa unavailable/failed: {exc}")
        return None


def _telea_inpaint(img_bgr: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """OpenCV Telea fallback inpainting."""
    return cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def step_inpainting(img_rgb: np.ndarray, blemish_mask: np.ndarray = None,
                    skin_mask: np.ndarray = None, inpaint_radius: int = None):
    """
    Inpainting with LaMa-first strategy.
      - Primary: LaMa (learned texture synthesis)
      - Fallback: OpenCV Telea
      - Mask is expanded + Gaussian-blurred for soft edges
      - Alpha-blend at edges for seamless transition
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, img_rgb, "Error: empty image."

        if inpaint_radius is None:
            inpaint_radius = DEFAULT_CONFIG["inpaint_radius"]

        backend = str(DEFAULT_CONFIG.get("inpaint_backend", "lama")).lower()
        if backend in {"mat", "zits", "sd", "sd_inpaint", "stable-diffusion"}:
            backend = "lama"
        if backend not in {"lama", "telea"}:
            backend = "lama"
        print(f"[Step 5] Inpainting — backend={backend} (r={inpaint_radius})")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # Generate blemish mask if not provided
        if blemish_mask is None:
            blemish_mask, _, _ = step_blemish_detection(img_rgb, skin_mask)
        if len(blemish_mask.shape) == 3:
            blemish_mask = cv2.cvtColor(blemish_mask, cv2.COLOR_RGB2GRAY)
        blemish_mask = cv2.resize(blemish_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Expand mask (controlled dilation + blur)
        expanded_mask, expand_info = step_expand_mask(blemish_mask, skin_mask)
        print(f"  {expand_info}")

        if np.count_nonzero(expanded_mask) == 0:
            print("  No blemish pixels to inpaint.")
            _save_debug("inpainted.jpg", img_bgr)
            return img_rgb, img_rgb, "No blemish pixels to inpaint."

        # Feather mask edges for smoother inpainting transition
        feathered = cv2.GaussianBlur(expanded_mask, (5, 5), 1.2)
        inp_mask = (feathered > 32).astype(np.uint8) * 255

        # Selected inpainting backend, then fallback to Telea.
        lama_rgb = None
        if backend == "lama":
            lama_rgb = _lama_inpaint(img_rgb, inp_mask)
        if lama_rgb is not None:
            result = cv2.cvtColor(lama_rgb, cv2.COLOR_RGB2BGR)
            inpaint_method = "lama"
        else:
            result = _telea_inpaint(img_bgr, inp_mask, inpaint_radius)
            inpaint_method = "telea"

        # Alpha-blend at soft edges for seamless transition
        alpha = cv2.GaussianBlur(expanded_mask.astype(np.float32) / 255.0, (9, 9), 0)
        alpha_3 = cv2.merge([alpha, alpha, alpha])
        result = np.clip(
            result.astype(np.float64) * alpha_3
            + img_bgr.astype(np.float64) * (1.0 - alpha_3),
            0, 255
        ).astype(np.uint8)

        # Pass 2: cleanup residual tiny red spots after first inpaint.
        lab_res = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        a_res = lab_res[:, :, 1].astype(np.float32)
        gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        lap_res = np.abs(cv2.Laplacian(gray_res, cv2.CV_32F))
        if skin_mask is not None:
            if len(skin_mask.shape) == 3:
                skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
            skin_resized = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            skin_resized = np.full((h, w), 255, dtype=np.uint8)
        skin_gate = (skin_resized > 127)
        a_skin = a_res[skin_gate]
        a_thr = (float(np.mean(a_skin)) + 0.9 * float(np.std(a_skin))) if a_skin.size else 136.0
        lap_thr = float(np.percentile(lap_res[skin_gate], 68)) if np.any(skin_gate) else 10.0
        residual = ((a_res > a_thr) & (lap_res > lap_thr) & skin_gate).astype(np.uint8) * 255
        residual = cv2.morphologyEx(
            residual,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        residual = _keep_small_components(
            residual,
            min_area=3,
            max_area=DEFAULT_CONFIG["micro_spot_max_area"],
        )
        residual_px = int(np.count_nonzero(residual))
        if residual_px > 0:
            second_radius = DEFAULT_CONFIG["second_inpaint_radius"]
            result2 = _telea_inpaint(result, residual, second_radius)
            alpha2 = cv2.GaussianBlur(residual.astype(np.float32) / 255.0, (7, 7), 0)
            alpha2_3 = cv2.merge([alpha2, alpha2, alpha2])
            result = np.clip(
                result2.astype(np.float64) * alpha2_3 + result.astype(np.float64) * (1.0 - alpha2_3),
                0,
                255,
            ).astype(np.uint8)

        _save_debug("inpainted.jpg", result)
        _save_debug("inpaint_mask_used.png", inp_mask)

        n_px = np.count_nonzero(inp_mask)
        info = (f"Inpainting complete.\n"
                f"Mask pixels: {n_px}\n"
                f"Radius: {inpaint_radius}\n"
                f"Method: {inpaint_method}\n"
                f"Residual pass pixels: {residual_px}")
        print(f"[Step 5] Done — {n_px} px, method={inpaint_method}, radius={inpaint_radius}")
        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(np.hstack([img_bgr, result]), cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 5] ERROR: {exc}")
        return img_rgb, img_rgb, f"Inpainting error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Multi-Scale Skin Smoothing — frequency separation + adaptive
# ═══════════════════════════════════════════════════════════════════════════
def step_redness_correction(img_rgb: np.ndarray, skin_mask: np.ndarray = None,
                            blemish_mask: np.ndarray = None):
    """Neutralize acne redness in LAB before smoothing."""
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, img_rgb, "Error: empty image."

        print("[Step 7] Redness Correction — LAB A-channel neutralization")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        if skin_mask is None:
            skin_mask = np.full((h, w), 255, dtype=np.uint8)
        if len(skin_mask.shape) == 3:
            skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
        skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if blemish_mask is not None:
            if len(blemish_mask.shape) == 3:
                blemish_mask = cv2.cvtColor(blemish_mask, cv2.COLOR_RGB2GRAY)
            blemish_mask = cv2.resize(blemish_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            target_mask = cv2.bitwise_and(skin_mask, blemish_mask)
        else:
            target_mask = skin_mask

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1].astype(np.float32)

        # Build redness gate from skin statistics (adaptive per face).
        skin_vals = a_channel[skin_mask > 127]
        skin_mean = float(np.mean(skin_vals)) if skin_vals.size else 128.0
        skin_std = float(np.std(skin_vals)) if skin_vals.size else 5.0
        gate_sigma = DEFAULT_CONFIG["redness_gate_sigma"]
        red_gate = (a_channel > (skin_mean + gate_sigma * skin_std)).astype(np.uint8) * 255

        # Restrict correction to skin + redness gate, but keep strong coverage around blemishes.
        target_soft = cv2.GaussianBlur(target_mask, (13, 13), 0)
        target_soft = (target_soft > 16).astype(np.uint8) * 255
        corr_mask = cv2.bitwise_or(
            cv2.bitwise_and(skin_mask, red_gate),
            cv2.bitwise_and(skin_mask, target_soft),
        )
        corr_f = cv2.GaussianBlur(corr_mask.astype(np.float32) / 255.0, (15, 15), 0)
        corr_f = corr_f[:, :, None]

        # Local target tone for A-channel (slightly toward skin_mean, not fully gray).
        a_local = cv2.GaussianBlur(a_channel, (31, 31), 0)
        neutral_target = 0.72 * a_local + 0.28 * skin_mean
        strength = DEFAULT_CONFIG["redness_strength"]
        a_corrected = a_channel * (1.0 - strength * corr_f[:, :, 0]) + neutral_target * (strength * corr_f[:, :, 0])

        # Residual redness cleanup on full skin (very mild, avoids flat/plastic look).
        residual_gate = (a_corrected > (skin_mean + 0.45 * skin_std)).astype(np.float32)
        residual_gate = cv2.GaussianBlur(residual_gate, (17, 17), 0)
        residual_gate = residual_gate * (skin_mask.astype(np.float32) / 255.0)
        global_s = DEFAULT_CONFIG["global_redness_strength"]
        a_final = a_corrected - global_s * residual_gate * np.maximum(a_corrected - skin_mean, 0.0)

        # Final selective anti-red pass on remaining red-biased skin only.
        post_s = DEFAULT_CONFIG["post_redness_pass_strength"]
        post_gate = ((a_final > (skin_mean + 0.25 * skin_std)) & (skin_mask > 127)).astype(np.float32)
        post_gate = cv2.GaussianBlur(post_gate, (19, 19), 0)
        a_final = a_final - post_s * post_gate * np.maximum(a_final - skin_mean, 0.0)

        lab[:, :, 1] = np.clip(a_final, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        _save_debug("redness_corrected.jpg", result)

        comparison = np.hstack([img_bgr, result])
        px = int(np.count_nonzero(corr_mask > 127))
        info = (
            f"Redness correction complete. Corrected pixels: {px}\n"
            f"skin_mean_A={skin_mean:.1f}, skin_std_A={skin_std:.1f}, strength={strength:.2f}"
        )
        print(f"[Step 7] Done — corrected pixels={px}, A_mean={skin_mean:.1f}, A_std={skin_std:.1f}")
        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 7] ERROR: {exc}")
        return img_rgb, img_rgb, f"Redness correction error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Multi-Scale Skin Smoothing — frequency separation + adaptive
# ═══════════════════════════════════════════════════════════════════════════
def _build_feature_mask_from_landmarks(img_rgb: np.ndarray):
    """Build a protection mask for eyes, brows, lips, nostrils from landmarks."""
    try:
        h, w = img_rgb.shape[:2]
        pts, _msg = _mediapipe_landmark_points(img_rgb)
        if pts is None:
            return None
        return _build_feature_protect_mask_from_pts(pts, h, w)

    except Exception as exc:
        print(f"  Feature mask from landmarks failed: {exc}")
        return None


def step_skin_retouch(img_rgb: np.ndarray, skin_mask: np.ndarray = None,
                      smooth_strength: int = None, texture_weight: float = None,
                      blemish_pct: float = None):
    """
    Guided-filter smoothing with frequency separation:
      1. low = GaussianBlur(image, sigma)
      2. high = image - low
      3. smooth_low = guidedFilter(low) ONLY inside skin mask
      4. result = smooth_low + high * texture_weight
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, img_rgb, "Error: empty image."

        if smooth_strength is None:
            smooth_strength = DEFAULT_CONFIG["smooth_strength"]
        if texture_weight is None:
            texture_weight = DEFAULT_CONFIG["texture_strength"]
        if blemish_pct is not None:
            # Low acne coverage => preserve more native pore texture.
            if blemish_pct < 0.3:
                texture_weight = max(texture_weight, 0.88)
            elif blemish_pct < 1.0:
                texture_weight = max(texture_weight, 0.84)
        guided_r = DEFAULT_CONFIG["guided_radius"]
        guided_eps = DEFAULT_CONFIG["guided_eps"]

        print("[Step 8] Skin Smoothing - guided filter")
        print(f"  guided_r={guided_r}, guided_eps={guided_eps}, "
              f"texture_w={texture_weight}")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        if skin_mask is None:
            skin_mask = np.full((h, w), 255, dtype=np.uint8)
        if len(skin_mask.shape) == 3:
            skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
        skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # ── Build feature protection mask ─────────────────────────
        protect_mask = _build_feature_mask_from_landmarks(img_rgb)
        if protect_mask is None:
            no_smooth_path = DEBUG_DIR / "no_smooth_mask.png"
            if no_smooth_path.exists():
                protect_mask = cv2.imread(str(no_smooth_path), cv2.IMREAD_GRAYSCALE)
                if protect_mask is not None:
                    protect_mask = cv2.resize(protect_mask, (w, h),
                                              interpolation=cv2.INTER_NEAREST)

        if protect_mask is not None:
            skin_mask = cv2.subtract(skin_mask, protect_mask)
            print("  Excluded eyes/brows/lips/nostrils (landmark-based).")

        # ── Edge-aware smoothing with guided filter ─────────────
        img_f = img_bgr.astype(np.float32) / 255.0

        # Guided filter: edge-preserving smooth (replaces Gaussian + freq-sep)
        smooth = cv2.ximgproc.guidedFilter(
            guide=img_f, src=img_f, radius=guided_r, eps=guided_eps
        )

        # Preserve texture via weighted blend
        result_f = smooth * (1 - texture_weight) + img_f * texture_weight
        retouched = np.clip(result_f * 255.0, 0, 255).astype(np.uint8)

        # ── Apply only within skin mask (soft edge) ───────────────
        mask_blur = cv2.GaussianBlur(skin_mask.astype(np.float32), (9, 9), 0) / 255.0
        mask_3 = cv2.merge([mask_blur, mask_blur, mask_blur])
        result = np.clip(
            retouched.astype(np.float64) * mask_3
            + img_bgr.astype(np.float64) * (1.0 - mask_3),
            0, 255,
        ).astype(np.uint8)

        _save_debug("skin_retouched.jpg", result)

        comparison = np.hstack([img_bgr, result])
        info = (f"Smoothing complete.\n"
                f"Mode: guided filter (edge-aware)\n"
                f"guided_r={guided_r}, guided_eps={guided_eps}, "
                f"texture_w={texture_weight:.2f}")
        print(f"[Step 8] Done - guided filter, r={guided_r}, "
              f"texture={texture_weight}")
        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 8] ERROR: {exc}")
        return img_rgb, img_rgb, f"Skin smoothing error: {exc}"


def step_tone_unify(img_rgb: np.ndarray, skin_mask: np.ndarray = None):
    """
    Evoto-like finishing pass:
    - homogenize low-frequency skin tone
    - keep pores/edges using detail mask
    - gently roll off strong highlights
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, img_rgb, "Error: empty image."

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        if skin_mask is None:
            skin_mask = np.full((h, w), 255, dtype=np.uint8)
        if len(skin_mask.shape) == 3:
            skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
        skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        strength = float(DEFAULT_CONFIG["tone_unify_strength"])
        radius = int(DEFAULT_CONFIG["tone_unify_radius"])
        highlight_rolloff = float(DEFAULT_CONFIG["highlight_rolloff"])

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # Low-frequency cleanup in chroma channels to remove patchy redness.
        A_s = cv2.bilateralFilter(A, d=radius, sigmaColor=18, sigmaSpace=16)
        B_s = cv2.bilateralFilter(B, d=radius, sigmaColor=14, sigmaSpace=14)

        # Keep high-frequency detail by reducing effect near strong gradients.
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edge = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        edge_n = cv2.normalize(edge, None, 0, 1, cv2.NORM_MINMAX)
        detail_guard = 1.0 - np.clip(edge_n * 1.6, 0.0, 1.0)

        skin_f = cv2.GaussianBlur((skin_mask > 127).astype(np.float32), (11, 11), 0)
        alpha = np.clip(skin_f * detail_guard * strength, 0.0, 1.0)

        A_out = A * (1.0 - alpha) + A_s * alpha
        B_out = B * (1.0 - alpha) + B_s * alpha

        # Highlight rolloff on skin (forehead/nose shine control).
        L_blur = cv2.GaussianBlur(L, (0, 0), 2.0)
        high = np.clip((L - L_blur) / 32.0, 0.0, 1.0)
        L_out = L - (highlight_rolloff * 25.0) * high * skin_f

        lab_out = np.dstack([np.clip(L_out, 0, 255), np.clip(A_out, 0, 255), np.clip(B_out, 0, 255)]).astype(np.uint8)
        out_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

        _save_debug("tone_unify.jpg", out_bgr)
        info = (
            f"Tone unify complete.\n"
            f"strength={strength:.2f}, radius={radius}, highlight_rolloff={highlight_rolloff:.2f}"
        )
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(np.hstack([img_bgr, out_bgr]), cv2.COLOR_BGR2RGB), info
    except Exception as exc:
        return img_rgb, img_rgb, f"Tone unify error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Texture Restore — clamped high-pass, exclude blemish areas
# ═══════════════════════════════════════════════════════════════════════════
def step_texture_restore(img_rgb: np.ndarray, sharpen_amount: float = None,
                         blemish_mask: np.ndarray = None):
    """
    Clamped high-pass texture restore:
      highpass = image - GaussianBlur(image, sigma=2)
      highpass = clip(highpass, -15, +15)
      result   = image + sharpen_amount * highpass
    Excludes blemish areas to avoid re-introducing removed spots.
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, img_rgb, "Error: empty image."

        if sharpen_amount is None:
            sharpen_amount = DEFAULT_CONFIG["sharpen_strength"]

        clamp_val = 12.0
        print(f"[Step 9] Texture Restore — amount={sharpen_amount}, clamp=±{clamp_val:.0f}")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        img_f = img_bgr.astype(np.float64)

        # High-pass: original − Gaussian(sigma=3)
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=3.0)
        high_pass = img_f - blurred

        # Clamp to ±15
        high_pass_clamped = np.clip(high_pass, -clamp_val, clamp_val)

        # Apply
        restored_f = img_f + sharpen_amount * high_pass_clamped

        # ── Exclude blemish areas from texture restoration ────────
        if blemish_mask is not None:
            if len(blemish_mask.shape) == 3:
                blemish_mask = cv2.cvtColor(blemish_mask, cv2.COLOR_RGB2GRAY)
            blemish_mask = cv2.resize(blemish_mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            blemish_dilated = cv2.dilate(blemish_mask, kernel, iterations=1)
            blemish_3 = cv2.merge([blemish_dilated] * 3).astype(np.float64) / 255.0
            restored_f = restored_f * (1.0 - blemish_3) + img_f * blemish_3
            print("  Excluded blemish areas from texture restoration.")

        result = np.clip(restored_f, 0, 255).astype(np.uint8)

        _save_debug("texture_restored.jpg", result)
        _save_debug("debug_final.jpg", result)

        comparison = np.hstack([img_bgr, result])
        info = (f"Texture restore complete.\n"
                f"Amount={sharpen_amount:.2f}, clamp=±{clamp_val:.0f}\n"
                f"Mode: clamped high-pass (sigma=2)")
        print(f"[Step 9] Done — amount={sharpen_amount}, clamp=±{clamp_val:.0f}")
        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 9] ERROR: {exc}")
        return img_rgb, img_rgb, f"Texture restore error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: Face Restoration — GFPGAN
# ═══════════════════════════════════════════════════════════════════════════
_gfpgan_restorer = None
_GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/"
    "download/v1.3.0/GFPGANv1.4.pth"
)
_GFPGAN_PATH = MODELS_DIR / "face_restore" / "GFPGANv1.4.pth"


def _download_gfpgan(model_path=_GFPGAN_PATH):
    """Download GFPGANv1.4 weights if not present."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [GFPGAN] Downloading model to {model_path} …")
    import urllib.request
    urllib.request.urlretrieve(_GFPGAN_URL, str(model_path))
    print(f"  [GFPGAN] Download complete ({model_path.stat().st_size / 1e6:.1f} MB)")
    return model_path


def _get_gfpgan():
    """Lazy-load GFPGAN model, auto-download if needed."""
    global _gfpgan_restorer
    if _gfpgan_restorer is not None:
        return _gfpgan_restorer
    model_path = _download_gfpgan()
    from gfpgan import GFPGANer
    _gfpgan_restorer = GFPGANer(
        model_path=str(model_path),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device="cpu",
    )
    print(f"  [GFPGAN] Model loaded from {model_path}")
    return _gfpgan_restorer


def step_face_restore(img_rgb: np.ndarray, blend: float = None, method: str = None):
    """
    AI face restoration using GFPGAN.
    Returns (result_rgb, comparison_rgb, info_str).
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, img_rgb, "Error: empty image."

        if blend is None:
            blend = DEFAULT_CONFIG["face_restore_blend"]

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        method = (method or DEFAULT_CONFIG.get("face_restore_method", "auto")).lower()
        codeformer_fidelity = float(DEFAULT_CONFIG.get("codeformer_fidelity", 0.7))
        print(
            f"[Step 10] Face Restoration — method={method}, "
            f"blend={blend:.2f}, fidelity={codeformer_fidelity:.2f}"
        )

        restored_rgb, info = restore_face_backend(
            img_rgb,
            blend=blend,
            method=method,
            codeformer_fidelity=codeformer_fidelity,
        )
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

        _save_debug("face_restored.jpg", restored_bgr)

        comparison = np.hstack([img_bgr, restored_bgr])
        print(f"[Step 10] Done — {method}")
        return (restored_rgb,
                cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 10] ERROR: {exc}")
        import traceback; traceback.print_exc()
        return img_rgb, img_rgb, f"Face restoration error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def run_full_pipeline(img_rgb: np.ndarray,
                      smooth_strength: int = None,
                      texture_weight: float = None,
                      sharpen: float = None,
                      blemish_threshold: float = None,
                      restore_blend: float = None,
                      tone_unify_enabled: bool = True,
                      inpaint_backend: str = "lama",
                      face_restore_method: str = "auto",
                      codeformer_fidelity: float = 0.7):
    """Run all 10 steps sequentially, save final result and comparison."""
    if img_rgb is None:
        return [None] * 10 + ["Please upload an image first."]

    if smooth_strength is None:
        smooth_strength = DEFAULT_CONFIG["smooth_strength"]
    if texture_weight is None:
        texture_weight = DEFAULT_CONFIG["texture_strength"]
    if sharpen is None:
        sharpen = DEFAULT_CONFIG["sharpen_strength"]
    if blemish_threshold is None:
        blemish_threshold = DEFAULT_CONFIG["blemish_ai_threshold"]
    if restore_blend is None:
        restore_blend = DEFAULT_CONFIG["face_restore_blend"]
    restore_blend = float(np.clip(restore_blend, 0.0, 1.0))
    DEFAULT_CONFIG["inpaint_backend"] = inpaint_backend
    DEFAULT_CONFIG["face_restore_method"] = face_restore_method
    DEFAULT_CONFIG["codeformer_fidelity"] = float(np.clip(codeformer_fidelity, 0.0, 1.0))

    log = []
    model_source = _resolve_blemish_model_source()
    log.append(f"[Model] Blemish AI source: {model_source}")
    print("=" * 60)
    print("FULL PIPELINE — START")
    print("=" * 60)
    print(f"[Model] Blemish AI source: {model_source}")

    # 1. Face Detection
    det_img, det_info = step_face_detection(img_rgb)
    log.append(f"[Step 1] {det_info}")

    # 2. Landmarks
    lm_img, lm_info = step_landmarks(img_rgb)
    log.append(f"[Step 2] {lm_info}")

    # 3. Face Parsing
    parse_mask, skin_mask, parse_info = step_face_parsing(img_rgb)
    log.append(f"[Step 3] Face parsing OK")

    # 4. Blemish Detection (uses skin mask)
    blemish_mask, blemish_overlay, blem_info = step_blemish_detection(
        img_rgb, skin_mask, blemish_threshold)
    log.append(f"[Step 4] {blem_info}")
    skin_pixels = max(int(np.count_nonzero(skin_mask > 127)), 1)
    blemish_pct = float(np.count_nonzero(blemish_mask > 0)) / skin_pixels * 100.0

    # 5. Expand blemish mask (adaptive per-component)
    expanded_mask, expand_info = step_expand_mask(blemish_mask, skin_mask)
    log.append(f"[Step 5] {expand_info}")

    # 6. Acne Inpainting
    inpainted, _, inp_info = step_inpainting(img_rgb, blemish_mask, skin_mask)
    log.append(f"[Step 6] {inp_info}")

    # 7. Redness correction
    red_corrected, _, red_info = step_redness_correction(inpainted, skin_mask, blemish_mask)
    log.append(f"[Step 7] {red_info}")

    # 7b. Optional learned skin-tone harmonizer (model-based)
    if DEFAULT_CONFIG.get("use_learned_tone_harmonizer", False) and harmonize_skin_tone_model is not None:
        red_corrected, harm_info = harmonize_skin_tone_model(
            red_corrected,
            skin_mask=skin_mask,
            strength=DEFAULT_CONFIG["learned_tone_strength"],
        )
        log.append(f"[Step 7b] {harm_info}")
    else:
        log.append("[Step 7b] Learned tone harmonizer skipped")

    # 8. Skin smoothing (edge-aware guided filter)
    retouched, _, ret_info = step_skin_retouch(
        red_corrected, skin_mask, smooth_strength, texture_weight, blemish_pct)
    log.append(f"[Step 8] {ret_info}")

    # 9. Tone unify — Evoto-like even chroma (optional if posterization on rare images)
    if tone_unify_enabled:
        unified, _, uni_info = step_tone_unify(retouched, skin_mask)
        log.append(f"[Step 9] {uni_info}")
    else:
        unified = retouched
        log.append("[Step 9] Tone unify skipped (checkbox off)")

    # 10. Texture Restore (clamped high-pass, excludes blemish areas)
    # Guardrail: high sharpen creates harsh/waxy artifacts.
    if blemish_pct >= 3.0:
        sharpen_cap = 0.08
    elif blemish_pct >= 1.5:
        sharpen_cap = 0.10
    else:
        sharpen_cap = 0.12
    used_sharpen = min(float(sharpen), sharpen_cap)
    if used_sharpen < float(sharpen):
        log.append(
            f"[Step 10] Texture amount clipped from {float(sharpen):.2f} to {used_sharpen:.2f} (mask={blemish_pct:.2f}%)"
        )
    restored, _, tex_info = step_texture_restore(unified, used_sharpen, blemish_mask)
    log.append(f"[Step 10] {tex_info}")

    # 11. Face Restoration (GFPGAN — adaptive cap by acne coverage)
    if restore_blend > 0:
        if blemish_pct >= 4.0:
            blend_cap = 0.35
        elif blemish_pct >= 2.0:
            blend_cap = 0.30
        else:
            blend_cap = 0.22
        used_blend = min(restore_blend, blend_cap)
        if used_blend < restore_blend:
            log.append(
                f"[Step 11] GFPGAN blend clipped from {restore_blend:.2f} to {used_blend:.2f} (mask={blemish_pct:.2f}%)"
            )
        face_restored, _, fr_info = step_face_restore(
            restored,
            used_blend,
            method=face_restore_method,
        )
        log.append(f"[Step 11] {fr_info}")
    else:
        face_restored = restored
        log.append("[Step 11] GFPGAN skipped (blend=0)")
        print("[Step 11] GFPGAN skipped (blend=0, damages skin texture)")

    # Save final result
    final = face_restored
    comparison_rgb = final
    try:
        final_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUTPUTS_DIR / "final_result.jpg"), final_bgr)
        print(f"Final result saved to outputs/final_result.jpg")

        # Side-by-side comparison: original | final
        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        comparison = np.hstack([orig_bgr, final_bgr])
        cv2.imwrite(str(OUTPUTS_DIR / "comparison.jpg"), comparison)
        comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        print(f"Comparison saved to outputs/comparison.jpg")
    except Exception as exc:
        log.append(f"[Save] Error saving final: {exc}")

    print("=" * 60)
    print("FULL PIPELINE — COMPLETE")
    print("=" * 60)

    return (det_img, lm_img, parse_mask, blemish_overlay,
            inpainted, red_corrected, retouched, restored, face_restored, comparison_rgb,
            "\n".join(log))


def apply_acne_heavy_preset(img_rgb: np.ndarray):
    """
    Auto-preset for acne-heavy portraits.
    Returns tuned slider values:
      (smooth_sigma, texture_weight, sharpen_amount, blemish_threshold, gfpgan_blend, log_msg)
    """
    # Safe defaults when no image is present.
    if img_rgb is None or img_rgb.size == 0:
        return 8, 0.90, 0.05, 0.30, 0.05, "Preset: upload image first, using safe acne-heavy defaults."

    try:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        # Loose skin proxy for fast severity estimation.
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_proxy = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 135)).astype(np.uint8)
        if np.count_nonzero(skin_proxy) < 500:
            skin_proxy = np.ones(img_rgb.shape[:2], dtype=np.uint8)

        a_ch = lab[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        a_vals = a_ch[skin_proxy == 1]
        lap_vals = lap[skin_proxy == 1]
        a_mean = float(np.mean(a_vals)) if a_vals.size else 128.0
        a_std = float(np.std(a_vals)) if a_vals.size else 5.0
        lap_p70 = float(np.percentile(lap_vals, 70)) if lap_vals.size else 12.0

        red_ratio = float(np.mean(a_vals > (a_mean + 0.8 * a_std))) if a_vals.size else 0.0
        tex_ratio = float(np.mean(lap_vals > lap_p70)) if lap_vals.size else 0.0
        severity = 0.65 * red_ratio + 0.35 * tex_ratio

        if severity > 0.33:
            # Heavy acne / diffuse redness
            smooth_sigma, texture_w, sharpen, blemish_t, gfpgan_b = 10, 0.91, 0.04, 0.28, 0.0
            level = "heavy"
        elif severity > 0.22:
            # Moderate acne
            smooth_sigma, texture_w, sharpen, blemish_t, gfpgan_b = 9, 0.90, 0.05, 0.30, 0.05
            level = "moderate"
        else:
            # Mild acne / mostly redness
            smooth_sigma, texture_w, sharpen, blemish_t, gfpgan_b = 8, 0.88, 0.06, 0.34, 0.08
            level = "mild"

        msg = (
            f"Applied Acne-heavy preset ({level}). "
            f"severity={severity:.3f}, red_ratio={red_ratio:.3f}, tex_ratio={tex_ratio:.3f}"
        )
        return smooth_sigma, texture_w, sharpen, blemish_t, gfpgan_b, msg
    except Exception as exc:
        return 8, 0.90, 0.05, 0.30, 0.05, f"Preset fallback used due to analysis error: {exc}"


def apply_profile_preset(profile_name: str, img_rgb: np.ndarray):
    """
    Apply profile preset with light adaptive adjustment from image severity.
    Returns:
      (smooth_sigma, texture_weight, sharpen_amount, blemish_threshold, gfpgan_blend, log_msg)
    """
    base = PROFILE_PRESETS.get(profile_name, PROFILE_PRESETS["Natural"]).copy()
    if img_rgb is None or img_rgb.size == 0:
        return (
            base["smooth"], base["texture"], base["sharpen"], base["blemish"], base["restore"],
            f"Applied profile '{profile_name}' with default values (no image loaded).",
        )
    try:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_proxy = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 135)).astype(np.uint8)
        if np.count_nonzero(skin_proxy) < 500:
            skin_proxy = np.ones(img_rgb.shape[:2], dtype=np.uint8)
        a_ch = lab[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        a_vals = a_ch[skin_proxy == 1]
        lap_vals = lap[skin_proxy == 1]
        a_mean = float(np.mean(a_vals)) if a_vals.size else 128.0
        a_std = float(np.std(a_vals)) if a_vals.size else 5.0
        lap_p70 = float(np.percentile(lap_vals, 70)) if lap_vals.size else 12.0
        red_ratio = float(np.mean(a_vals > (a_mean + 0.85 * a_std))) if a_vals.size else 0.0
        tex_ratio = float(np.mean(lap_vals > lap_p70)) if lap_vals.size else 0.0
        severity = 0.7 * red_ratio + 0.3 * tex_ratio

        if severity > 0.33:
            base["blemish"] = max(0.22, base["blemish"] - 0.03)
            base["smooth"] = min(14, base["smooth"] + 1)
            if profile_name == "Evoto Target":
                base["texture"] = max(0.70, base["texture"] - 0.04)
                base["restore"] = float(min(0.22, max(base["restore"], 0.12)))
            else:
                base["restore"] = min(base["restore"], 0.03)
        elif severity < 0.18:
            base["blemish"] = min(0.5, base["blemish"] + 0.03)
            base["smooth"] = max(5, base["smooth"] - 1)
            base["texture"] = min(0.95, base["texture"] + 0.01)

        msg = (
            f"Applied profile '{profile_name}' (severity={severity:.3f}, "
            f"red={red_ratio:.3f}, tex={tex_ratio:.3f})."
        )
        return base["smooth"], base["texture"], base["sharpen"], base["blemish"], base["restore"], msg
    except Exception as exc:
        return (
            base["smooth"], base["texture"], base["sharpen"], base["blemish"], base["restore"],
            f"Applied profile '{profile_name}' with fallback due to analysis error: {exc}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI — Evoto-inspired (dark workspace, before/after slider, export)
# ═══════════════════════════════════════════════════════════════════════════
EVOTO_CSS = """
.gradio-container { max-width: 1320px !important; margin-left: auto !important; margin-right: auto !important; }
.evoto-topbar {
    display: flex !important;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    background: linear-gradient(180deg, #2d2d32 0%, #222226 100%);
    border: 1px solid #3d3d44;
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
.evoto-brand-wrap { display: flex; align-items: center; gap: 10px; }
.evoto-brand-mark {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg, #f7d54a, #e8a317);
    box-shadow: 0 0 12px rgba(247, 213, 74, 0.45);
}
.evoto-brand-title { font-size: 1.15rem !important; font-weight: 700 !important; color: #f4f4f5 !important; margin: 0 !important; letter-spacing: -0.02em; }
.evoto-brand-sub { font-size: 0.78rem !important; color: #9ca3af !important; margin: 0 !important; }
.evoto-top-actions { display: flex !important; align-items: center; gap: 8px; flex-wrap: wrap; }
.evoto-bottombar {
    display: flex !important;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    margin-top: 12px;
    background: #252529;
    border: 1px solid #35353c;
    border-radius: 12px;
}
.evoto-side-panel {
    padding: 14px !important;
    background: linear-gradient(180deg, #26262b, #1c1c20) !important;
    border: 1px solid #35353d !important;
    border-radius: 14px !important;
    min-height: 520px;
}
.evoto-main-panel {
    padding: 12px !important;
    background: #1a1a1e !important;
    border: 1px solid #2e2e34 !important;
    border-radius: 14px !important;
}
.evoto-export-wrap button, .evoto-export-wrap a {
    background: linear-gradient(180deg, #fce07a, #e5a917) !important;
    color: #1a1a1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    min-width: 108px;
}
.evoto-export-wrap button:hover, .evoto-export-wrap a:hover { filter: brightness(1.06); }
.evoto-run-primary { min-height: 48px !important; font-weight: 700 !important; }
"""


def get_evoto_theme():
    """Gradio 6: pass to launch(theme=...), not Blocks(...)."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.zinc,
    ).set(
        body_background_fill="#121214",
        body_background_fill_dark="#121214",
        block_background_fill="#1c1c20",
        block_background_fill_dark="#1c1c20",
        block_label_text_color="#e4e4e7",
        block_title_text_color="#fafafa",
        input_background_fill="#27272a",
        border_color_primary="#3f3f46",
        button_primary_text_color="#18181b",
    )


def run_full_pipeline_for_ui(
    img_rgb: np.ndarray,
    smooth_strength: int,
    texture_weight: float,
    sharpen: float,
    blemish_threshold: float,
    restore_blend: float,
    tone_unify_enabled: bool,
    inpaint_backend: str,
    face_restore_method: str,
    codeformer_fidelity: float,
):
    """Chạy pipeline + cặp ảnh cho ImageSlider + đường dẫn file Xuất + state cho realtime."""
    empty_slider = (None, None)
    empty_state = None
    if img_rgb is None:
        z = None
        return (
            z, z, z, z, z, z, z, z, z, z,
            "Vui lòng tải ảnh chân dung lên.",
            empty_slider,
            None,
            empty_state,
        )
    (
        det_img,
        lm_img,
        parse_mask,
        blemish_overlay,
        inpainted,
        red_corrected,
        retouched,
        restored,
        face_restored,
        comparison_rgb,
        log,
    ) = run_full_pipeline(
        img_rgb,
        smooth_strength,
        texture_weight,
        sharpen,
        blemish_threshold,
        restore_blend,
        tone_unify_enabled,
        inpaint_backend,
        face_restore_method,
        codeformer_fidelity,
    )
    final_path = OUTPUTS_DIR / "final_result.jpg"
    dl = str(final_path.resolve()) if final_path.is_file() else None
    state = (img_rgb.copy(), face_restored.copy())
    return (
        det_img,
        lm_img,
        parse_mask,
        blemish_overlay,
        inpainted,
        red_corrected,
        retouched,
        restored,
        face_restored,
        comparison_rgb,
        log,
        (img_rgb, face_restored),
        dl,
        state,
    )


def apply_display_realtime(
    state: tuple[np.ndarray, np.ndarray] | None,
    strength: float,
    brightness: float,
    contrast: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend + chỉnh hiển thị. Trả về (orig, hiển_thị) cho ImageSlider."""
    if state is None:
        return (None, None)
    orig, final = state
    s = np.clip(strength / 100.0, 0.0, 1.0)
    blended = (
        orig.astype(np.float32) * (1 - s)
        + final.astype(np.float32) * s
    ).clip(0, 255).astype(np.uint8)
    if brightness != 0 or contrast != 1.0:
        lab = cv2.cvtColor(blended, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip((lab[:, :, 0] - 128) * contrast + 128 + brightness, 0, 255)
        blended = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return (orig, blended)


def build_ui():
    with gr.Blocks(title="Retouch AI") as demo:
        with gr.Row(elem_classes="evoto-topbar"):
            gr.HTML(
                '<div class="evoto-brand-wrap"><span class="evoto-brand-mark"></span>'
                '<div><p class="evoto-brand-title">Retouch AI</p>'
                '<p class="evoto-brand-sub">Trước / sau · Xuất ảnh · Phong cách Evoto</p></div></div>'
            )
            with gr.Row(elem_classes="evoto-top-actions"):
                gr.Button("Hoàn tác", size="sm", interactive=False)
                gr.Button("Làm lại", size="sm", interactive=False)
                btn_export = gr.DownloadButton(
                    "Xuất",
                    value=None,
                    elem_classes="evoto-export-wrap",
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=300, elem_classes="evoto-side-panel"):
                input_img = gr.Image(label="Ảnh gốc", type="numpy", height=300)
                dd_profile = gr.Dropdown(
                    choices=list(PROFILE_PRESETS.keys()),
                    value="Evoto Target",
                    label="Chất lượng mục tiêu",
                )
                with gr.Row():
                    btn_profile = gr.Button("Áp preset", variant="secondary", scale=1)
                    btn_preset = gr.Button("Mụn nặng", variant="secondary", scale=1)
                cb_tone_unify = gr.Checkbox(
                    value=True,
                    label="Đồng nhất tone da (bước 9)",
                )
                with gr.Accordion("Điều chỉnh chi tiết", open=True):
                    sl_smooth = gr.Slider(
                        1, 20, DEFAULT_CONFIG["smooth_strength"], step=1, label="Độ làm mịn"
                    )
                    sl_texture = gr.Slider(
                        0.0, 1.0, DEFAULT_CONFIG["texture_strength"], step=0.05, label="Giữ chi tiết da"
                    )
                    sl_sharpen = gr.Slider(
                        0.0, 0.5, DEFAULT_CONFIG["sharpen_strength"], step=0.05, label="Khôi phục texture"
                    )
                    sl_blemish = gr.Slider(
                        0.1, 0.9, DEFAULT_CONFIG["blemish_ai_threshold"], step=0.05, label="Ngưỡng AI (mụn)"
                    )
                    sl_restore_blend = gr.Slider(
                        0.0, 1.0, DEFAULT_CONFIG["face_restore_blend"], step=0.1, label="GFPGAN (0–1)"
                    )
                    dd_inpaint_backend = gr.Dropdown(
                        choices=["lama", "telea", "mat", "zits", "sd"],
                        value=DEFAULT_CONFIG["inpaint_backend"],
                        label="Inpaint backend",
                    )
                    dd_face_restore = gr.Dropdown(
                        choices=["auto", "gfpgan", "codeformer"],
                        value=DEFAULT_CONFIG["face_restore_method"],
                        label="Face restore backend",
                    )
                    sl_codeformer_fidelity = gr.Slider(
                        0.0,
                        1.0,
                        DEFAULT_CONFIG["codeformer_fidelity"],
                        step=0.05,
                        label="CodeFormer fidelity",
                    )
                btn_all = gr.Button(
                    "Áp dụng retouch",
                    variant="primary",
                    size="lg",
                    elem_classes="evoto-run-primary",
                )

            with gr.Column(scale=11, elem_classes="evoto-main-panel"):
                blend_state = gr.State(None)
                with gr.Row():
                    sl_strength = gr.Slider(0, 100, 100, step=1, label="Cường độ retouch")
                    sl_brightness = gr.Slider(-30, 30, 0, step=1, label="Độ sáng")
                    sl_contrast = gr.Slider(0.7, 1.5, 1.0, step=0.05, label="Độ tương phản")
                before_after = gr.ImageSlider(
                    label="Trước · Sau",
                    type="numpy",
                    slider_position=50,
                    max_height=600,
                )
                out_log = gr.Textbox(label="Nhật ký xử lý", lines=6, max_lines=16)

        with gr.Accordion("Xem từng bước (debug)", open=False):
            with gr.Row():
                out_det = gr.Image(label="1. Phát hiện mặt")
                out_lm = gr.Image(label="2. Landmark")
                out_parse = gr.Image(label="3. Parse")
                out_blem = gr.Image(label="4. Mặt nạ mụn")
            with gr.Row():
                out_inp = gr.Image(label="6. Inpaint")
                out_red = gr.Image(label="7. Hết đỏ")
                out_ret = gr.Image(label="8. Làm mịn")
                out_tex = gr.Image(label="9. Texture")
            with gr.Row():
                out_face_rest = gr.Image(label="10. GFPGAN")
                out_comp = gr.Image(label="Cạnh nhau")

        with gr.Row(elem_classes="evoto-bottombar"):
            gr.Button("Đồng bộ", size="sm", interactive=False)
            gr.Radio(
                ["Nữ", "Nam"],
                value="Nữ",
                label="Preset giới tính (giao diện)",
            )

        btn_preset.click(
            apply_acne_heavy_preset,
            inputs=[input_img],
            outputs=[sl_smooth, sl_texture, sl_sharpen, sl_blemish, sl_restore_blend, out_log],
        )
        btn_profile.click(
            apply_profile_preset,
            inputs=[dd_profile, input_img],
            outputs=[sl_smooth, sl_texture, sl_sharpen, sl_blemish, sl_restore_blend, out_log],
        )
        btn_all.click(
            run_full_pipeline_for_ui,
            inputs=[
                input_img,
                sl_smooth,
                sl_texture,
                sl_sharpen,
                sl_blemish,
                sl_restore_blend,
                cb_tone_unify,
                dd_inpaint_backend,
                dd_face_restore,
                sl_codeformer_fidelity,
            ],
            outputs=[
                out_det,
                out_lm,
                out_parse,
                out_blem,
                out_inp,
                out_red,
                out_ret,
                out_tex,
                out_face_rest,
                out_comp,
                out_log,
                before_after,
                btn_export,
                blend_state,
            ],
        )

        def _on_display_change(state, strength, brightness, contrast):
            return apply_display_realtime(state, strength, brightness, contrast)

        _display_inputs = [blend_state, sl_strength, sl_brightness, sl_contrast]
        for sl in (sl_strength, sl_brightness, sl_contrast):
            sl.release(
                _on_display_change,
                inputs=_display_inputs,
                outputs=[before_after],
                show_progress="hidden",
            )

        with gr.Accordion("Công cụ từng bước", open=False):
            gr.Markdown("Dùng chung ảnh **Ảnh gốc** ở cột trái.")
            with gr.Tabs():
                with gr.Tab("1. Phát hiện mặt"):
                    btn1 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o1_img = gr.Image(label="Kết quả")
                        o1_info = gr.Textbox(label="Thông tin", lines=4)
                    btn1.click(step_face_detection, [input_img], [o1_img, o1_info])

                with gr.Tab("2. Landmark"):
                    btn2 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o2_img = gr.Image(label="Kết quả")
                        o2_info = gr.Textbox(label="Thông tin", lines=2)
                    btn2.click(step_landmarks, [input_img], [o2_img, o2_info])

                with gr.Tab("3. Parse vùng mặt"):
                    btn3 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o3_mask = gr.Image(label="Mặt nạ phân vùng")
                        o3_skin = gr.Image(label="Da")
                    o3_info = gr.Textbox(label="Thông tin", lines=6)
                    btn3.click(step_face_parsing, [input_img], [o3_mask, o3_skin, o3_info])

                with gr.Tab("4. Phát hiện mụn"):
                    s4_thresh = gr.Slider(
                        0.1, 0.9, DEFAULT_CONFIG["blemish_ai_threshold"],
                        step=0.05, label="Ngưỡng AI",
                    )
                    btn4 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o4_mask = gr.Image(label="Mặt nạ")
                        o4_overlay = gr.Image(label="Overlay")
                    o4_info = gr.Textbox(label="Thông tin", lines=3)
                    btn4.click(
                        lambda img, t: step_blemish_detection(img, None, t),
                        [input_img, s4_thresh],
                        [o4_mask, o4_overlay, o4_info],
                    )

                with gr.Tab("5. Inpaint"):
                    btn5 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o5_img = gr.Image(label="Kết quả")
                        o5_comp = gr.Image(label="So sánh")
                    o5_info = gr.Textbox(label="Thông tin", lines=2)
                    btn5.click(
                        lambda img: step_inpainting(img),
                        [input_img],
                        [o5_img, o5_comp, o5_info],
                    )

                with gr.Tab("7. Hết đỏ"):
                    btn_red = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o_red = gr.Image(label="Kết quả")
                        o_red_comp = gr.Image(label="So sánh")
                    o_red_info = gr.Textbox(label="Thông tin", lines=2)
                    btn_red.click(
                        lambda img: step_redness_correction(img),
                        [input_img],
                        [o_red, o_red_comp, o_red_info],
                    )

                with gr.Tab("8. Làm mịn da"):
                    with gr.Row():
                        s6_smooth = gr.Slider(
                            1, 20, DEFAULT_CONFIG["smooth_strength"], step=1, label="Độ làm mịn"
                        )
                        s6_texture = gr.Slider(
                            0.0, 1.0, DEFAULT_CONFIG["texture_strength"],
                            step=0.05, label="Chi tiết da",
                        )
                    btn6 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o6_img = gr.Image(label="Kết quả")
                        o6_comp = gr.Image(label="So sánh")
                    o6_info = gr.Textbox(label="Thông tin", lines=2)
                    btn6.click(
                        lambda img, s, t: step_skin_retouch(img, None, s, t),
                        [input_img, s6_smooth, s6_texture],
                        [o6_img, o6_comp, o6_info],
                    )

                with gr.Tab("9. Texture"):
                    s7_sharp = gr.Slider(
                        0.0, 0.5, DEFAULT_CONFIG["sharpen_strength"],
                        step=0.05, label="Sharpen",
                    )
                    btn7 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o7_img = gr.Image(label="Kết quả")
                        o7_comp = gr.Image(label="So sánh")
                    o7_info = gr.Textbox(label="Thông tin", lines=2)
                    btn7.click(
                        lambda img, s: step_texture_restore(img, s),
                        [input_img, s7_sharp],
                        [o7_img, o7_comp, o7_info],
                    )

                with gr.Tab("10. GFPGAN"):
                    gr.Markdown("Khôi phục / làm net khuôn mặt (GFPGAN).")
                    with gr.Row():
                        s8_blend = gr.Slider(
                            0.0, 1.0, DEFAULT_CONFIG["face_restore_blend"],
                            step=0.1, label="Blend (0=gốc, 1=restored)",
                        )
                    btn8 = gr.Button("Chạy", variant="primary")
                    with gr.Row():
                        o8_img = gr.Image(label="Kết quả")
                        o8_comp = gr.Image(label="So sánh")
                    o8_info = gr.Textbox(label="Thông tin", lines=3)
                    btn8.click(
                        lambda img, b: step_face_restore(img, b),
                        [input_img, s8_blend],
                        [o8_img, o8_comp, o8_info],
                    )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=get_evoto_theme(),
        css=EVOTO_CSS,
    )
