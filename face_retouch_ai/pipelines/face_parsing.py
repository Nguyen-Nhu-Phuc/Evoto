"""
Step 3 — Face Parsing using BiSeNet (face-parsing.PyTorch).
Auto-downloads 79999_iter.pth from Google Drive if missing.
Returns skin mask and full segmentation map.
"""

import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models, transforms

DEBUG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debug"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "face_parsing"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_CKPT_PATH = MODELS_DIR / "79999_iter.pth"
_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"

# 19-class label map
LABEL_NAMES = {
    0: "background", 1: "skin", 2: "l_brow", 3: "r_brow",
    4: "l_eye", 5: "r_eye", 6: "eyeglass", 7: "l_ear", 8: "r_ear",
    9: "earring", 10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip",
    14: "neck", 15: "necklace", 16: "cloth", 17: "hair", 18: "hat",
}
EXCLUDE_LABELS = {2, 3, 4, 5, 10, 11, 12, 13, 17}

# ──────────────────────────── BiSeNet architecture ────────────────────────


class _ConvBNReLU(nn.Module):
    def __init__(self, ci, co, ks=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, ks, s, p, bias=False)
        self.bn = nn.BatchNorm2d(co)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _ARM(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = _ConvBNReLU(ci, co)
        self.conv_atten = nn.Conv2d(co, co, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(co)

    def forward(self, x):
        f = self.conv(x)
        a = torch.sigmoid(
            self.bn_atten(
                self.conv_atten(torch.mean(f, dim=[2, 3], keepdim=True))
            )
        )
        return f * a


class _ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        r = tv_models.resnet18(weights=None)
        self.conv1, self.bn1, self.relu = r.conv1, r.bn1, r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4
        self.arm16 = _ARM(256, 128)
        self.arm32 = _ARM(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128)
        self.conv_head16 = _ConvBNReLU(128, 128)
        self.conv_avg = _ConvBNReLU(512, 128, ks=1, s=1, p=0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        f8 = self.layer2(x)
        f16 = self.layer3(f8)
        f32 = self.layer4(f16)
        avg = self.conv_avg(torch.mean(f32, dim=[2, 3], keepdim=True))
        f32_up = self.conv_head32(
            self.arm32(f32)
            + F.interpolate(avg, size=f32.shape[2:], mode="nearest")
        )
        f32_up = F.interpolate(f32_up, size=f16.shape[2:], mode="nearest")
        f16_up = self.conv_head16(self.arm16(f16) + f32_up)
        f16_up = F.interpolate(f16_up, size=f8.shape[2:], mode="nearest")
        return f8, f16_up, f32_up


class _SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvBNReLU(3, 64, 7, 2, 3)
        self.conv2 = _ConvBNReLU(64, 64, 3, 2, 1)
        self.conv3 = _ConvBNReLU(64, 64, 3, 2, 1)
        self.conv_out = _ConvBNReLU(64, 128)

    def forward(self, x):
        return self.conv_out(self.conv3(self.conv2(self.conv1(x))))


class _FFM(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.convblk = _ConvBNReLU(ci, co, ks=1, s=1, p=0)
        self.conv1 = nn.Conv2d(co, co // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(co // 4, co, 1, bias=False)

    def forward(self, fsp, fcp):
        f = self.convblk(torch.cat([fsp, fcp], 1))
        a = torch.sigmoid(
            self.conv2(
                F.relu(self.conv1(torch.mean(f, dim=[2, 3], keepdim=True)))
            )
        )
        return f + f * a


class _Out(nn.Module):
    def __init__(self, ci, mi, nc):
        super().__init__()
        self.conv = _ConvBNReLU(ci, mi)
        self.conv_out = nn.Conv2d(mi, nc, 1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class BiSeNet(nn.Module):
    def __init__(self, nc=19):
        super().__init__()
        self.cp = _ContextPath()
        self.sp = _SpatialPath()
        self.ffm = _FFM(256, 256)
        self.conv_out = _Out(256, 256, nc)
        self.conv_out16 = _Out(128, 64, nc)
        self.conv_out32 = _Out(128, 64, nc)

    def forward(self, x):
        H, W = x.shape[2:]
        _f8, fcp8, _ = self.cp(x)
        fsp = self.sp(x)
        return F.interpolate(
            self.conv_out(self.ffm(fsp, fcp8)),
            (H, W),
            mode="bilinear",
            align_corners=True,
        )


# ──────────────────────────── Model loading ────────────────────────────

_cached_model = None


def _ensure_checkpoint():
    """Download 79999_iter.pth from Google Drive if not present."""
    if _CKPT_PATH.exists():
        return
    print(f"  Downloading BiSeNet checkpoint to {_CKPT_PATH} …")
    try:
        import gdown

        gdown.download(id=_GDRIVE_ID, output=str(_CKPT_PATH), quiet=False)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot download BiSeNet weights: {exc}\n"
            f"Download manually from Google Drive file id={_GDRIVE_ID} "
            f"and place at {_CKPT_PATH}"
        ) from exc


def _get_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    _ensure_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiSeNet(19).to(device).eval()

    sd = torch.load(str(_CKPT_PATH), map_location=device, weights_only=True)
    remapped = {}
    for k, v in sd.items():
        remapped[k.replace("cp.resnet.", "cp.")] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [BiSeNet] Missing keys: {len(missing)} (batch-norm buffers — OK)")
    if unexpected:
        print(f"  [BiSeNet] Unexpected keys: {len(unexpected)}")

    _cached_model = (model, device)
    return _cached_model


# ──────────────────────────── Public API ────────────────────────────────

_PREPROCESS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def parse_face(img_rgb: np.ndarray):
    """
    Run BiSeNet face parsing.

    Returns
    -------
    parsing_map : np.ndarray (H, W) uint8  — class index per pixel
    skin_mask   : np.ndarray (H, W) uint8  — 255 where skin
    colour_vis  : np.ndarray (H, W, 3) RGB — coloured segmentation overlay
    info        : str
    """
    print("[Step 3] Face Parsing — BiSeNet")
    h, w = img_rgb.shape[:2]
    model, device = _get_model()

    inp = _PREPROCESS(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

    # Resize parsing map back to original size
    parsing_full = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)

    # Skin mask (label 1)
    skin_raw = (parsing_full == 1).astype(np.uint8) * 255
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_raw, cv2.MORPH_OPEN, kern, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kern, iterations=2)

    # Subtract excluded regions
    for lbl in EXCLUDE_LABELS:
        skin_mask[parsing_full == lbl] = 0

    # Colour visualisation
    np.random.seed(42)
    palette = np.random.randint(0, 255, (20, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    colour_vis = palette[parsing_full]

    # Debug saves
    cv2.imwrite(str(DEBUG_DIR / "step3_skin_mask.png"), skin_mask)
    cv2.imwrite(str(DEBUG_DIR / "step3_parsing.jpg"), cv2.cvtColor(colour_vis, cv2.COLOR_RGB2BGR))

    # Stats
    unique, counts = np.unique(parsing_full, return_counts=True)
    info = "Detected regions:\n"
    for u, c in zip(unique, counts):
        pct = c / parsing_full.size * 100
        info += f"  {LABEL_NAMES.get(u, f'class_{u}')}: {pct:.1f}%\n"
    info += f"Skin mask: {np.count_nonzero(skin_mask)} px"

    print(f"[Step 3] Done — skin {np.count_nonzero(skin_mask)} px")
    return parsing_full, skin_mask, colour_vis, info
