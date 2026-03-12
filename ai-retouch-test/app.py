"""
app.py — AI Portrait Retouch Pipeline (v3)
============================================
Professional blemish-removal pipeline. Key principles:
  - NEVER modify eyes, eyebrows, lips, nose, hair
  - Preserve natural pores and skin texture
  - Targeted blemish removal (acne, red spots, pimples only)

Pipeline:
  1.  Face Detection    (RetinaFace → MediaPipe fallback)
  2.  Landmark Detection (MediaPipe FaceMesh 478)
  3.  Face Parsing      (BiSeNet — strict skin mask excluding features)
  4.  Blemish Detection (LAB redness + Laplacian texture, multi-condition)
  4b. Mask Expansion    (5×5 ellipse dilation + Gaussian blur edges)
  5.  Inpainting        (Telea radius 7, blurred soft mask)
  6.  Skin Smoothing    (guided filter on low-freq, inside skin mask only)
  7.  Texture Restore   (clamped high-pass sigma=2, ±15, amount=0.08)
"""

import sys
import cv2
import numpy as np
import gradio as gr
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEBUG_DIR = OUTPUTS_DIR / "debug"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Default configuration — easy to tune
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "smooth_strength": 10,      # Gaussian sigma for low-freq separation
    "texture_strength": 0.55,   # high-freq blend weight (1.0=original, 0=smooth)
    "sharpen_strength": 0.08,   # clamped high-pass texture restore
    "blemish_threshold": 140,   # A-channel floor (redness gate)
    "texture_threshold": 15,    # Laplacian texture anomaly threshold
    "blemish_max_area": 300,    # remove connected components larger than this
    "blemish_min_area": 5,      # remove connected components smaller than this
    "mask_dilate_size": 5,      # ellipse kernel size for mask expansion
    "inpaint_radius": 7,        # Telea inpainting radius
    "guided_radius": 9,         # guided filter radius for skin smoothing
    "guided_eps": 0.01,         # guided filter epsilon
}


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
    """Detect 468/478 facial landmarks. Ignore landmarks outside face bbox."""
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, "Error: empty image."

        print("[Step 2] Landmark Detection — start")
        import mediapipe as mp

        h, w = img_rgb.shape[:2]
        _lm_model = MODELS_DIR / "mediapipe" / "face_landmarker.task"
        if not _lm_model.exists():
            return img_rgb, f"Model file missing: {_lm_model}"

        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_lm_model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as landmarker:
            result = landmarker.detect(mp_img)

        if not result.face_landmarks:
            return img_rgb, "No landmarks detected."

        # Select exactly one face (first)
        face_lms = result.face_landmarks[0]
        n = len(face_lms)

        # Compute face bounding box from landmarks
        xs = [lm.x * w for lm in face_lms]
        ys = [lm.y * h for lm in face_lms]
        fx1, fy1 = int(min(xs)), int(min(ys))
        fx2, fy2 = int(max(xs)), int(max(ys))

        annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Draw mesh using tasks API drawing utils
        mp_draw = mp.tasks.vision.drawing_utils
        FLC = mp.tasks.vision.FaceLandmarksConnections
        mp_draw.draw_landmarks(
            annotated, face_lms, FLC.FACE_LANDMARKS_TESSELATION,
        )
        mp_draw.draw_landmarks(
            annotated, face_lms, FLC.FACE_LANDMARKS_CONTOURS,
        )

        # Draw individual points — skip landmarks outside face bbox
        drawn = 0
        for lm in face_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                cv2.circle(annotated, (cx, cy), 1, (0, 255, 0), -1)
                drawn += 1

        _save_debug("landmarks.jpg", annotated)
        result_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        info = f"Detected {n} landmarks ({drawn} inside face bbox)."
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
    Multi-condition blemish detection:
      1. LAB A-channel redness:  red_score = A - 128 ; red_mask = A > floor
      2. Laplacian texture anomaly: texture_mask = |Laplacian(gray)| > thresh
      3. Combine:  blemish = red_mask AND texture_mask AND skin_mask
      4. Connected-component filter: remove area > max_area or < min_area
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return (np.zeros((1, 1), dtype=np.uint8),
                    img_rgb, "Error: empty image.")

        a_floor = threshold if threshold is not None else DEFAULT_CONFIG["blemish_threshold"]
        tex_thresh = DEFAULT_CONFIG["texture_threshold"]
        min_area = DEFAULT_CONFIG["blemish_min_area"]
        max_area = DEFAULT_CONFIG["blemish_max_area"]

        print(f"[Step 4] Blemish Detection — LAB redness + Laplacian")
        print(f"  A_floor={a_floor}, texture_thresh={tex_thresh}")
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

        # ── Condition 1: LAB A-channel redness ────────────────────
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        A_ch = lab[:, :, 1].astype(np.float32)
        red_score = A_ch - 128.0
        red_mask = (A_ch > a_floor).astype(np.uint8)

        skin_a = A_ch[skin_bin == 1]
        print(f"  A-channel on skin: mean={np.mean(skin_a):.1f}, "
              f"std={np.std(skin_a):.1f}, "
              f"red_mask pixels={np.count_nonzero(red_mask & skin_bin)}")

        # ── Condition 2: Laplacian texture anomaly ────────────────
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32)
        texture_mask = (lap > tex_thresh).astype(np.uint8)

        print(f"  Laplacian: mean={np.mean(lap[skin_bin == 1]):.1f}, "
              f"max={np.max(lap[skin_bin == 1]):.0f}, "
              f"texture_mask pixels={np.count_nonzero(texture_mask & skin_bin)}")

        # ── Combine: both conditions must be true, inside skin ────
        mask = (red_mask & texture_mask & skin_bin) * 255

        # ── Morphological cleanup ─────────────────────────────────
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # ── Connected-component filtering ─────────────────────────
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        removed_small = 0
        removed_large = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                mask[labels == i] = 0
                removed_small += 1
            elif area > max_area:
                mask[labels == i] = 0
                removed_large += 1
        print(f"  Removed {removed_small} tiny (< {min_area} px), "
              f"{removed_large} large (> {max_area} px) regions")

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

        info = (f"Blemish pixels: {n_blemish}\n"
                f"% of skin area: {pct_skin:.2f}%\n"
                f"% of total image: {pct_total:.2f}%\n"
                f"A_floor={a_floor}, texture_thresh={tex_thresh}")
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
        print(f"[Step 4b] Mask Expansion — ellipse {ks}×{ks}, 1 iter + blur")
        h, w = blemish_mask.shape[:2]
        original_count = np.count_nonzero(blemish_mask)

        if len(blemish_mask.shape) == 3:
            blemish_mask = cv2.cvtColor(blemish_mask, cv2.COLOR_RGB2GRAY)

        # Step 1: controlled dilation — single pass, small kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        expanded = cv2.dilate(blemish_mask, kernel, iterations=1)

        # Step 2: blur edges for soft inpainting transition
        expanded_blur = cv2.GaussianBlur(expanded, (9, 9), 0)
        expanded = (expanded_blur > 32).astype(np.uint8) * 255

        # Step 3: constrain to skin region
        if skin_mask is not None:
            if len(skin_mask.shape) == 3:
                skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
            skin_resized = cv2.resize(skin_mask, (w, h),
                                       interpolation=cv2.INTER_NEAREST)
            expanded = cv2.bitwise_and(expanded, expanded,
                                        mask=(skin_resized > 127).astype(np.uint8) * 255)

        expanded_count = np.count_nonzero(expanded)
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
def step_inpainting(img_rgb: np.ndarray, blemish_mask: np.ndarray = None,
                    skin_mask: np.ndarray = None, inpaint_radius: int = None):
    """
    Inpainting with Telea algorithm.
      - radius=7 (stronger fill)
      - Mask is expanded + Gaussian-blurred for soft edges
      - Alpha-blend at edges for seamless transition
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, img_rgb, "Error: empty image."

        if inpaint_radius is None:
            inpaint_radius = DEFAULT_CONFIG["inpaint_radius"]

        print(f"[Step 5] Inpainting — Telea radius={inpaint_radius}")
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

        # Soft-edge mask via Gaussian blur for seamless blending
        mask_float = expanded_mask.astype(np.float32) / 255.0
        mask_blur = cv2.GaussianBlur(mask_float, (9, 9), 0)
        inp_mask = (mask_blur > 0.1).astype(np.uint8) * 255

        # Telea inpainting
        result = cv2.inpaint(img_bgr, inp_mask, inpaintRadius=inpaint_radius,
                             flags=cv2.INPAINT_TELEA)

        # Alpha-blend at soft edges for seamless transition
        alpha = cv2.GaussianBlur(expanded_mask.astype(np.float32) / 255.0, (9, 9), 0)
        alpha_3 = cv2.merge([alpha, alpha, alpha])
        result = np.clip(
            result.astype(np.float64) * alpha_3
            + img_bgr.astype(np.float64) * (1.0 - alpha_3),
            0, 255
        ).astype(np.uint8)

        _save_debug("inpainted.jpg", result)
        _save_debug("inpaint_mask_used.png", inp_mask)

        n_px = np.count_nonzero(inp_mask)
        info = (f"Inpainting complete.\n"
                f"Mask pixels: {n_px}\n"
                f"Radius: {inpaint_radius}")
        print(f"[Step 5] Done — {n_px} px, radius={inpaint_radius}")
        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(np.hstack([img_bgr, result]), cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 5] ERROR: {exc}")
        return img_rgb, img_rgb, f"Inpainting error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Multi-Scale Skin Smoothing — frequency separation + adaptive
# ═══════════════════════════════════════════════════════════════════════════
def _build_feature_mask_from_landmarks(img_rgb: np.ndarray):
    """Build a protection mask for eyes, brows, lips, nostrils from landmarks."""
    try:
        import mediapipe as mp
        h, w = img_rgb.shape[:2]

        _lm_model = MODELS_DIR / "mediapipe" / "face_landmarker.task"
        if not _lm_model.exists():
            return None

        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_lm_model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as landmarker:
            result = landmarker.detect(mp_img)

        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]
        pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in lms])

        protect_mask = np.zeros((h, w), dtype=np.uint8)

        # MediaPipe FaceMesh landmark index groups
        # Left eye
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Left eyebrow
        left_brow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        # Right eyebrow
        right_brow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        # Lips (outer)
        lips_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        # Nostrils
        nostrils = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275]

        for group in [left_eye, right_eye, left_brow, right_brow, lips_outer, nostrils]:
            group_pts = pts[group]
            hull = cv2.convexHull(group_pts)
            cv2.fillConvexPoly(protect_mask, hull, 255)

        # Dilate slightly to ensure full coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        protect_mask = cv2.dilate(protect_mask, kernel, iterations=1)

        return protect_mask

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
        guided_r = DEFAULT_CONFIG["guided_radius"]
        guided_eps = DEFAULT_CONFIG["guided_eps"]

        print(f"[Step 6] Skin Smoothing — guided filter")
        print(f"  sigma={smooth_strength}, texture_w={texture_weight}, "
              f"guided_r={guided_r}, guided_eps={guided_eps}")
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

        # ── Frequency separation ──────────────────────────────────
        img_f = img_bgr.astype(np.float32) / 255.0

        # Low frequency: Gaussian captures skin tone
        low = cv2.GaussianBlur(img_f, (0, 0), sigmaX=smooth_strength)

        # High frequency: texture details (pores, fine lines)
        high = img_f - low

        # ── Guided filter on low-freq ONLY inside skin mask ───────
        smooth_low = cv2.ximgproc.guidedFilter(
            guide=low, src=low, radius=guided_r, eps=guided_eps
        )

        # Recombine: smoothed low + original high * weight
        result_f = smooth_low + high * texture_weight
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
                f"Mode: guided filter + freq-sep\n"
                f"sigma={smooth_strength}, guided_r={guided_r}, "
                f"texture_w={texture_weight:.2f}")
        print(f"[Step 6] Done — guided filter, sigma={smooth_strength}, "
              f"texture={texture_weight}")
        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 6] ERROR: {exc}")
        return img_rgb, img_rgb, f"Skin smoothing error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Texture Restore — clamped high-pass, exclude blemish areas
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

        clamp_val = 15.0
        print(f"[Step 7] Texture Restore — amount={sharpen_amount}, clamp=±{clamp_val:.0f}")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        img_f = img_bgr.astype(np.float64)

        # High-pass: original − Gaussian(sigma=2)
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=2.0)
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
        print(f"[Step 7] Done — amount={sharpen_amount}, clamp=±{clamp_val:.0f}")
        return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), info)

    except Exception as exc:
        print(f"[Step 7] ERROR: {exc}")
        return img_rgb, img_rgb, f"Texture restore error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def run_full_pipeline(img_rgb: np.ndarray,
                      smooth_strength: int = None,
                      texture_weight: float = None,
                      sharpen: float = None,
                      blemish_threshold: float = None):
    """Run all 7 steps sequentially, save final result and comparison."""
    if img_rgb is None:
        return [None] * 8 + ["Please upload an image first."]

    if smooth_strength is None:
        smooth_strength = DEFAULT_CONFIG["smooth_strength"]
    if texture_weight is None:
        texture_weight = DEFAULT_CONFIG["texture_strength"]
    if sharpen is None:
        sharpen = DEFAULT_CONFIG["sharpen_strength"]
    if blemish_threshold is None:
        blemish_threshold = DEFAULT_CONFIG["blemish_threshold"]

    log = []
    print("=" * 60)
    print("FULL PIPELINE — START")
    print("=" * 60)

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

    # 4b. Expand blemish mask (adaptive per-component)
    expanded_mask, expand_info = step_expand_mask(blemish_mask, skin_mask)
    log.append(f"[Step 4b] {expand_info}")

    # 5. Edge-Aware Inpainting (blurred mask, radius 3)
    inpainted, _, inp_info = step_inpainting(img_rgb, blemish_mask, skin_mask)
    log.append(f"[Step 5] {inp_info}")

    # 6. Skin Smoothing (frequency separation, sigma=8, high*0.8)
    retouched, _, ret_info = step_skin_retouch(
        inpainted, skin_mask, smooth_strength, texture_weight)
    log.append(f"[Step 6] {ret_info}")

    # 7. Texture Restore (clamped high-pass, excludes blemish areas)
    restored, _, tex_info = step_texture_restore(retouched, sharpen, blemish_mask)
    log.append(f"[Step 7] {tex_info}")

    # Save final result
    comparison_rgb = restored
    try:
        final_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
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
            inpainted, retouched, restored, comparison_rgb,
            "\n".join(log))


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════
def build_ui():
    with gr.Blocks(title="AI Portrait Retouch Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# AI Portrait Retouch Pipeline\n"
            "Upload a portrait and run individual steps or the full pipeline."
        )

        with gr.Row():
            input_img = gr.Image(label="Upload portrait", type="numpy")

        # ── Tab: Full Pipeline ─────────────────────────────────────
        with gr.Tab("Full Pipeline"):
            gr.Markdown("Run all 7 steps in sequence.")
            with gr.Row():
                sl_smooth = gr.Slider(1, 20, DEFAULT_CONFIG["smooth_strength"],
                                      step=1, label="Smooth Sigma")
                sl_texture = gr.Slider(0.0, 1.0, DEFAULT_CONFIG["texture_strength"],
                                       step=0.05, label="Texture Weight")
                sl_sharpen = gr.Slider(0.0, 0.5, DEFAULT_CONFIG["sharpen_strength"],
                                       step=0.05, label="Sharpen Amount")
                sl_blemish = gr.Slider(125, 150, DEFAULT_CONFIG["blemish_threshold"],
                                       step=1, label="A-channel Floor")
            btn_all = gr.Button("Run Full Pipeline", variant="primary", size="lg")

            with gr.Row():
                out_det = gr.Image(label="1. Face Detection")
                out_lm = gr.Image(label="2. Landmarks")
                out_parse = gr.Image(label="3. Face Parsing")
                out_blem = gr.Image(label="4. Blemish")
            with gr.Row():
                out_inp = gr.Image(label="5. Inpainted")
                out_ret = gr.Image(label="6. Skin Retouch")
                out_tex = gr.Image(label="7. Texture Restore")
                out_comp = gr.Image(label="Original | Final")
            out_log = gr.Textbox(label="Log", lines=10)

            btn_all.click(
                run_full_pipeline,
                inputs=[input_img, sl_smooth, sl_texture, sl_sharpen, sl_blemish],
                outputs=[out_det, out_lm, out_parse, out_blem,
                         out_inp, out_ret, out_tex, out_comp, out_log],
            )

        # ── Tab: Step 1 ───────────────────────────────────────────
        with gr.Tab("1. Face Detection"):
            btn1 = gr.Button("Detect faces", variant="primary")
            with gr.Row():
                o1_img = gr.Image(label="Result")
                o1_info = gr.Textbox(label="Info", lines=4)
            btn1.click(step_face_detection, [input_img], [o1_img, o1_info])

        # ── Tab: Step 2 ───────────────────────────────────────────
        with gr.Tab("2. Landmarks"):
            btn2 = gr.Button("Detect landmarks", variant="primary")
            with gr.Row():
                o2_img = gr.Image(label="Result")
                o2_info = gr.Textbox(label="Info", lines=2)
            btn2.click(step_landmarks, [input_img], [o2_img, o2_info])

        # ── Tab: Step 3 ───────────────────────────────────────────
        with gr.Tab("3. Face Parsing"):
            btn3 = gr.Button("Parse face regions", variant="primary")
            with gr.Row():
                o3_mask = gr.Image(label="Segmentation Mask")
                o3_skin = gr.Image(label="Skin Mask")
            o3_info = gr.Textbox(label="Info", lines=6)
            btn3.click(step_face_parsing, [input_img], [o3_mask, o3_skin, o3_info])

        # ── Tab: Step 4 ───────────────────────────────────────────
        with gr.Tab("4. Blemish Detection"):
            s4_thresh = gr.Slider(125, 150, DEFAULT_CONFIG["blemish_threshold"],
                                  step=1, label="A-channel Floor")
            btn4 = gr.Button("Detect blemishes", variant="primary")
            with gr.Row():
                o4_mask = gr.Image(label="Blemish Mask")
                o4_overlay = gr.Image(label="Overlay")
            o4_info = gr.Textbox(label="Info", lines=3)
            btn4.click(
                lambda img, t: step_blemish_detection(img, None, t),
                [input_img, s4_thresh], [o4_mask, o4_overlay, o4_info],
            )

        # ── Tab: Step 5 ───────────────────────────────────────────
        with gr.Tab("5. Inpainting"):
            btn5 = gr.Button("Inpaint blemishes", variant="primary")
            with gr.Row():
                o5_img = gr.Image(label="Result")
                o5_comp = gr.Image(label="Comparison")
            o5_info = gr.Textbox(label="Info", lines=2)
            btn5.click(
                lambda img: step_inpainting(img),
                [input_img], [o5_img, o5_comp, o5_info],
            )

        # ── Tab: Step 6 ───────────────────────────────────────────
        with gr.Tab("6. Skin Retouch"):
            with gr.Row():
                s6_smooth = gr.Slider(1, 20, DEFAULT_CONFIG["smooth_strength"],
                                      step=1, label="Smooth Sigma")
                s6_texture = gr.Slider(0.0, 1.0, DEFAULT_CONFIG["texture_strength"],
                                       step=0.05, label="Texture Weight")
            btn6 = gr.Button("Smooth skin", variant="primary")
            with gr.Row():
                o6_img = gr.Image(label="Result")
                o6_comp = gr.Image(label="Comparison")
            o6_info = gr.Textbox(label="Info", lines=2)
            btn6.click(
                lambda img, s, t: step_skin_retouch(img, None, s, t),
                [input_img, s6_smooth, s6_texture], [o6_img, o6_comp, o6_info],
            )

        # ── Tab: Step 7 ───────────────────────────────────────────
        with gr.Tab("7. Texture Restore"):
            s7_sharp = gr.Slider(0.0, 0.5, DEFAULT_CONFIG["sharpen_strength"],
                                 step=0.05, label="Sharpen Amount")
            btn7 = gr.Button("Restore texture", variant="primary")
            with gr.Row():
                o7_img = gr.Image(label="Result")
                o7_comp = gr.Image(label="Comparison")
            o7_info = gr.Textbox(label="Info", lines=2)
            btn7.click(
                lambda img, s: step_texture_restore(img, s),
                [input_img, s7_sharp], [o7_img, o7_comp, o7_info],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
