"""
Learned skin-tone harmonization (optional).

If checkpoint exists at models/skin_tone/harmonizer.pth, this module predicts
a gentle RGB residual to unify skin tone while preserving structure.
If unavailable, caller should fall back to classic color correction.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "skin_tone"
CKPT_PATH = MODELS_DIR / "harmonizer.pth"
_cached_model = None


class _ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SkinToneHarmonizer(nn.Module):
    """Small residual CNN: input RGB+mask -> residual RGB."""
    def __init__(self, base=32):
        super().__init__()
        self.e1 = _ConvBlock(4, base)
        self.e2 = _ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.mid = _ConvBlock(base * 2, base * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = _ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        m = self.mid(e2)
        d = self.up(m)
        d = self.d1(torch.cat([d, e1], dim=1))
        return torch.tanh(self.out(d))


def _get_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not CKPT_PATH.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinToneHarmonizer().to(device).eval()
    try:
        weights = torch.load(str(CKPT_PATH), map_location=device, weights_only=True)
    except TypeError:
        weights = torch.load(str(CKPT_PATH), map_location=device)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    model.load_state_dict(weights, strict=False)
    _cached_model = (model, device)
    print(f"  [ToneHarmonizer] Loaded model from {CKPT_PATH}")
    return _cached_model


def harmonize_skin_tone_model(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray | None = None,
    strength: float = 0.35,
) -> tuple[np.ndarray, str]:
    """
    Apply learned skin-tone harmonization if model exists.
    Returns (result_rgb, info_str). Falls back to original when unavailable.
    """
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb, "Tone harmonizer: empty image."
    model_info = _get_model()
    if model_info is None:
        return img_rgb, "Tone harmonizer model unavailable (using classic pipeline)."

    model, device = model_info
    h, w = img_rgb.shape[:2]
    if skin_mask is None:
        skin_mask = np.full((h, w), 255, dtype=np.uint8)
    if len(skin_mask.shape) == 3:
        skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
    skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    skin_f = (skin_mask.astype(np.float32) / 255.0)[None, None, :, :]

    x = img_rgb.astype(np.float32) / 255.0
    x_t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    m_t = torch.from_numpy(skin_f).to(device)
    inp = torch.cat([x_t, m_t], dim=1)

    with torch.no_grad():
        residual = model(inp).clamp(-1.0, 1.0) * 0.15
        out = (x_t + residual * float(strength)).clamp(0.0, 1.0)
    out_rgb = (out[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

    # Blend inside skin only.
    skin_3 = np.repeat((skin_mask.astype(np.float32) / 255.0)[:, :, None], 3, axis=2)
    result = (
        img_rgb.astype(np.float32) * (1.0 - skin_3)
        + out_rgb.astype(np.float32) * skin_3
    ).clip(0, 255).astype(np.uint8)
    return result, f"Tone harmonizer applied (strength={strength:.2f})."


def harmonize_skin_tone_classic(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray | None = None,
    strength: float = 0.25,
) -> tuple[np.ndarray, str]:
    """
    Classic skin-tone harmonization using local LAB statistics.

    Idea: within skin, pull pixel (a,b) slightly toward the robust mean (median)
    to reduce patchy colour shifts while preserving luminance/shape.
    """
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb, "Tone harmonizer (classic): empty image."
    h, w = img_rgb.shape[:2]
    if skin_mask is None:
        skin_mask = np.full((h, w), 255, dtype=np.uint8)
    if len(skin_mask.shape) == 3:
        skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
    skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    m = (skin_mask > 0)
    if int(np.count_nonzero(m)) < 50:
        return img_rgb, "Tone harmonizer (classic): skin mask too small."

    a_med = float(np.median(a[m]))
    b_med = float(np.median(b[m]))

    # Per-pixel strength: stronger inside skin, but keep gentle overall.
    s = float(np.clip(strength, 0.0, 1.0))
    a_new = a + (a_med - a) * s
    b_new = b + (b_med - b) * s
    lab[:, :, 1] = np.where(m, a_new, a)
    lab[:, :, 2] = np.where(m, b_new, b)

    out = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out, f"Tone harmonizer classic applied (strength={s:.2f})."


def harmonize_skin_tone(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray | None = None,
    strength: float = 0.30,
) -> tuple[np.ndarray, str]:
    """Use learned model if present, else classic fallback."""
    out, info = harmonize_skin_tone_model(img_rgb, skin_mask=skin_mask, strength=strength)
    if "unavailable" in info.lower():
        return harmonize_skin_tone_classic(img_rgb, skin_mask=skin_mask, strength=min(0.35, strength))
    return out, info

