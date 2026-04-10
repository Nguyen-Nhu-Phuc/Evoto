"""
ModelScope portrait skin retouching: iic/cv_unet_skin-retouching (Alibaba DAMO).

Requires: pip install modelscope tensorflow
The upstream SkinRetouchingPipeline also downloads damo/cv_resnet50_face-detection_retinaface
on first use (face crops).

Weights: place under models/modelscope/iic_cv_unet_skin_retouching/ or run:
  modelscope download --model iic/cv_unet_skin-retouching --local_dir models/modelscope/iic_cv_unet_skin_retouching

Gradio gives RGB ndarray; ModelScope LoadImage flips channels for ndarray only — we pass PIL
like the web demo so colors stay correct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MS_LOCAL_DIR = MODELS_DIR / "modelscope" / "iic_cv_unet_skin_retouching"
MODEL_ID = "iic/cv_unet_skin-retouching"

_pipeline: Any = None
_pipeline_device: str | None = None


def resolve_modelscope_skin_dir() -> Path | None:
    """Return local model dir if complete, else None."""
    need = ("pytorch_model.pt", "joint_20210926.pth", "tf_graph.pb")
    if not MS_LOCAL_DIR.is_dir():
        return None
    if all((MS_LOCAL_DIR / n).is_file() for n in need):
        return MS_LOCAL_DIR
    return None


def download_modelscope_skin(local_dir: Path | None = None) -> Path:
    """Download full model repo via ModelScope SDK."""
    dest = Path(local_dir) if local_dir is not None else MS_LOCAL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download

    snapshot_download(MODEL_ID, local_dir=str(dest))
    return dest


def _pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_pipeline(device: str | None = None):
    global _pipeline, _pipeline_device
    dev = _pick_device(device)
    if _pipeline is not None and _pipeline_device == dev:
        return _pipeline

    model_dir = resolve_modelscope_skin_dir()
    if model_dir is None:
        download_modelscope_skin()

    model_dir = resolve_modelscope_skin_dir()
    if model_dir is None:
        raise FileNotFoundError(
            f"ModelScope skin model incomplete under {MS_LOCAL_DIR}. "
            f"Run: modelscope download --model {MODEL_ID} --local_dir {MS_LOCAL_DIR}"
        )

    try:
        import tensorflow as tf  # noqa: F401 — required by SkinRetouchingPipeline
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for iic/cv_unet_skin-retouching. "
            "Install: pip install tensorflow"
        ) from exc

    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline as ms_pipeline
    from modelscope.utils.constant import Tasks

    _pipeline_device = dev
    _pipeline = ms_pipeline(
        Tasks.skin_retouching,
        model=str(model_dir),
        device=dev,
    )
    return _pipeline


def _result_to_rgb_bgr(result: dict) -> np.ndarray:
    from modelscope.outputs import OutputKeys

    img = result.get(OutputKeys.OUTPUT_IMG)
    if img is None:
        img = result.get("pred")
    if img is None:
        raise ValueError("ModelScope skin retouch returned no image tensor.")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    # Pipeline returns BGR uint8 (see skin_retouching_pipeline forward).
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def apply_modelscope_skin_retouch(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray | None = None,
    blend: float = 1.0,
    device: str | None = None,
) -> tuple[np.ndarray, str]:
    """
    Run Alibaba U-Net skin retouching and optionally blend onto ``img_rgb`` using ``skin_mask``.

    blend: 0 = keep original, 1 = full retouched output on masked region (soft edges).
    """
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb, "Empty image; skipped ModelScope skin."

    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return img_rgb, "ModelScope skin blend=0; skipped."

    pipe = _get_pipeline(device=device)
    # ndarray triggers LoadImage.convert_to_ndarray → channel flip; PIL matches hub demo (RGB).
    pil_in = Image.fromarray(np.ascontiguousarray(img_rgb.astype(np.uint8)))
    result = pipe(pil_in)
    ms_rgb = _result_to_rgb_bgr(result)

    h, w = img_rgb.shape[:2]
    if ms_rgb.shape[:2] != (h, w):
        ms_rgb = cv2.resize(ms_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    if skin_mask is None or blend >= 0.999:
        return ms_rgb, f"ModelScope skin OK (full frame), shape={ms_rgb.shape}"

    if len(skin_mask.shape) == 3:
        skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
    m = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    m = cv2.GaussianBlur(m.astype(np.float32), (15, 15), 0) / 255.0
    m = np.clip(m * blend, 0.0, 1.0)
    m3 = np.stack([m, m, m], axis=-1)
    out = img_rgb.astype(np.float32) * (1.0 - m3) + ms_rgb.astype(np.float32) * m3
    return (
        np.clip(out, 0, 255).astype(np.uint8),
        f"ModelScope skin OK (skin blend={blend:.2f}), shape={ms_rgb.shape}",
    )


def modelscope_skin_available() -> tuple[bool, str]:
    """Quick check for UI / logs (no heavy TF import if modelscope missing)."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False, "tensorflow not installed (required for this model)"
    try:
        import modelscope  # noqa: F401
    except ImportError:
        return False, "modelscope not installed"
    d = resolve_modelscope_skin_dir()
    if d is None:
        return False, f"weights missing under {MS_LOCAL_DIR}"
    return True, f"ready ({d.name})"
