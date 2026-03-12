"""Check all models and dependencies for the pipeline."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 60)
print("  MODEL & DEPENDENCY STATUS")
print("=" * 60)

# 1. BiSeNet
bp = ROOT / "models" / "face_parsing" / "79999_iter.pth"
if bp.exists():
    sz = bp.stat().st_size / 1e6
    print(f"\n[1] BiSeNet (face parsing): OK ({sz:.1f} MB)")
else:
    print(f"\n[1] BiSeNet (face parsing): MISSING  ->  {bp}")

# 2. LaMa
lp = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt"
if lp.exists():
    sz = lp.stat().st_size / 1e6
    print(f"[2] LaMa (inpainting):     OK ({sz:.1f} MB) at {lp}")
else:
    print(f"[2] LaMa (inpainting):     MISSING (auto-downloads on first run)")

# 3. U-Net blemish
unet = ROOT / "models" / "blemish_seg" / "unet_blemish.pth"
if unet.exists():
    print(f"[3] U-Net blemish seg:     OK")
else:
    print(f"[3] U-Net blemish seg:     NOT TRAINED (heuristic fallback active)")

# 4. RetinaFace
try:
    from retinaface import RetinaFace
    print(f"[4] RetinaFace (serengil): OK")
except Exception:
    print(f"[4] RetinaFace (serengil): NOT INSTALLED (MediaPipe fallback active)")

# 5. MediaPipe
try:
    import mediapipe as mp
    print(f"[5] MediaPipe:             OK (v{mp.__version__})")
except Exception:
    print(f"[5] MediaPipe:             MISSING")

# 6. OpenCV contrib
try:
    import cv2
    _ = cv2.ximgproc.guidedFilter
    print(f"[6] OpenCV contrib:        OK (v{cv2.__version__}, guidedFilter available)")
except Exception:
    print(f"[6] OpenCV contrib:        MISSING (need opencv-contrib-python)")

# 7. PyTorch
try:
    import torch
    cuda = "CUDA" if torch.cuda.is_available() else "CPU only"
    print(f"[7] PyTorch:               OK (v{torch.__version__}, {cuda})")
except Exception:
    print(f"[7] PyTorch:               MISSING")

# 8. simple-lama-inpainting
try:
    from simple_lama_inpainting import SimpleLama
    print(f"[8] simple-lama-inpainting:OK")
except Exception:
    print(f"[8] simple-lama-inpainting:MISSING")

# 9. gdown
try:
    import gdown
    print(f"[9] gdown:                 OK")
except Exception:
    print(f"[9] gdown:                 MISSING")

print()
print("=" * 60)
print("  RUNTIME LOAD TEST")
print("=" * 60)

# Load BiSeNet
try:
    sys.path.insert(0, str(ROOT))
    from pipelines.face_parsing import _get_model
    model, device = _get_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  BiSeNet: Loaded OK — {n_params:,} params on {device}")
except Exception as e:
    print(f"\n  BiSeNet: FAILED — {e}")

# Load LaMa
try:
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
    print(f"  LaMa:    Loaded OK")
except Exception as e:
    print(f"  LaMa:    FAILED — {e}")

# MediaPipe face detection
try:
    import mediapipe as mp2
    det = mp2.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    det.close()
    print(f"  MediaPipe FaceDetect: OK")
except Exception as e:
    print(f"  MediaPipe FaceDetect: FAILED — {e}")

# MediaPipe face mesh
try:
    mesh = mp2.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                            refine_landmarks=True, min_detection_confidence=0.5)
    mesh.close()
    print(f"  MediaPipe FaceMesh:   OK")
except Exception as e:
    print(f"  MediaPipe FaceMesh:   FAILED — {e}")

print()
print("=" * 60)
all_ok = bp.exists() and lp.exists()
print(f"  VERDICT: {'ALL CORE MODELS OK' if all_ok else 'SOME MODELS MISSING'}")
print("=" * 60)
