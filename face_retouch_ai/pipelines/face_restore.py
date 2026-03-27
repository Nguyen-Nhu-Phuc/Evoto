"""
Step 9 — AI Face Restoration.

Supports:
  - GFPGAN (built-in)
  - CodeFormer (optional, via external repository script)
  - auto mode: try CodeFormer first, then GFPGAN
"""

import os
import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "face_restore"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/"
    "download/v1.3.0/GFPGANv1.4.pth"
)
_GFPGAN_PATH = MODELS_DIR / "GFPGANv1.4.pth"
_CODEFORMER_REPO = MODELS_DIR / "codeformer"

_cached_restorer = None


def download_gfpgan(model_path: Path = _GFPGAN_PATH) -> Path | None:
    """Download GFPGANv1.4 weights if not already present."""
    try:
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  [GFPGAN] Downloading model to {model_path} …")
        import urllib.request
        urllib.request.urlretrieve(_GFPGAN_URL, str(model_path))
        print(f"  [GFPGAN] Download complete ({model_path.stat().st_size / 1e6:.1f} MB)")
        return model_path

    except Exception as e:
        print(f"  [GFPGAN] Download error: {e}")
        return None


def load_gfpgan(model_path: Path = _GFPGAN_PATH):
    """
    Load GFPGAN model. Returns GFPGANer instance or None on failure.
    Caches the model after first load.
    """
    global _cached_restorer
    try:
        if _cached_restorer is not None:
            return _cached_restorer

        if model_path is None or not model_path.exists():
            model_path = download_gfpgan(model_path or _GFPGAN_PATH)
            if model_path is None:
                return None

        from gfpgan import GFPGANer

        restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cpu",
        )
        _cached_restorer = restorer
        print(f"  [GFPGAN] Model loaded from {model_path}")
        return restorer

    except Exception as e:
        print(f"  [GFPGAN] Load error: {e}")
        return None


def _run_codeformer_external(
    img_rgb: np.ndarray,
    fidelity: float = 0.7,
) -> np.ndarray | None:
    """
    Run CodeFormer via its official inference script if available locally.

    Expected layout:
      models/face_restore/codeformer/inference_codeformer.py
    """
    script = _CODEFORMER_REPO / "inference_codeformer.py"
    if not script.exists():
        return None

    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    with tempfile.TemporaryDirectory(prefix="codeformer_") as tmp_dir:
        tmp = Path(tmp_dir)
        inp_dir = tmp / "inputs"
        out_dir = tmp / "results"
        inp_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        input_path = inp_dir / "face.png"
        cv2.imwrite(str(input_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        cmd = [
            "python",
            str(script),
            "-i",
            str(inp_dir),
            "-o",
            str(out_dir),
            "--fidelity_weight",
            f"{fidelity:.3f}",
            "--bg_upsampler",
            "none",
        ]
        try:
            run = subprocess.run(
                cmd,
                cwd=str(_CODEFORMER_REPO),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=180,
                check=False,
            )
            if run.returncode != 0:
                print(f"  [CodeFormer] Script failed: {run.stderr[-500:]}")
                return None
        except Exception as exc:
            print(f"  [CodeFormer] Execution error: {exc}")
            return None

        restored_path = None
        candidates = list((out_dir / "final_results").glob("*.png")) + list(
            (out_dir / "final_results").glob("*.jpg")
        )
        if candidates:
            restored_path = candidates[0]
        if restored_path is None:
            flat = list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))
            if flat:
                restored_path = flat[0]
        if restored_path is None:
            print("  [CodeFormer] Output not found.")
            return None

        restored_bgr = cv2.imread(str(restored_path))
        if restored_bgr is None:
            return None
        if restored_bgr.shape[:2] != img_rgb.shape[:2]:
            restored_bgr = cv2.resize(
                restored_bgr,
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)


def restore_face(
    img_rgb: np.ndarray,
    blend: float = 0.20,
    blemish_mask: np.ndarray = None,
    max_coverage_pct: float = 2.0,
    method: str = "auto",
    codeformer_fidelity: float = 0.7,
) -> tuple[np.ndarray, str]:
    """
    Restore face quality with selectable backend and safety limits.

    When acne coverage exceeds max_coverage_pct of the face,
    GFPGAN is skipped to avoid smearing artefacts over large areas.
    Default blend reduced to 0.20 for subtlety.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3) RGB uint8
    blend   : float 0–1 — 0 = original, 1 = fully restored
    blemish_mask : np.ndarray optional — if provided, check coverage
    max_coverage_pct : float — skip GFPGAN if coverage > this %

    method : {"auto","gfpgan","codeformer"}
        - auto: try CodeFormer first, then GFPGAN fallback
        - gfpgan: force GFPGAN
        - codeformer: force CodeFormer (no GFPGAN fallback)
    codeformer_fidelity : float in [0,1]
        CodeFormer fidelity_weight. Higher keeps more original identity/detail.

    Returns
    -------
    (result_rgb, info_str)
    """
    try:
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb, "Error: empty image."

        # Check acne coverage — skip GFPGAN for heavy acne faces
        if blemish_mask is not None:
            total_px = blemish_mask.shape[0] * blemish_mask.shape[1]
            blemish_px = np.count_nonzero(blemish_mask)
            coverage = (blemish_px / max(total_px, 1)) * 100
            if coverage > max_coverage_pct:
                print(f"[Step 9] GFPGAN SKIPPED — acne coverage {coverage:.1f}% > {max_coverage_pct}%")
                return img_rgb, f"GFPGAN skipped (coverage {coverage:.1f}% > {max_coverage_pct}% limit)"

        method = (method or "auto").strip().lower()
        if method not in {"auto", "gfpgan", "codeformer"}:
            method = "auto"

        print(f"[Step 9] Face Restoration — method={method}, blend={blend:.2f}")

        # Try CodeFormer first (auto / forced)
        if method in {"auto", "codeformer"}:
            codeformer_rgb = _run_codeformer_external(
                img_rgb,
                fidelity=codeformer_fidelity,
            )
            if codeformer_rgb is not None:
                if blend < 1.0:
                    codeformer_rgb = cv2.addWeighted(
                        codeformer_rgb, blend, img_rgb, 1.0 - blend, 0
                    )
                info = (
                    "CodeFormer restoration complete. "
                    f"Blend={blend:.2f}, fidelity={codeformer_fidelity:.2f}"
                )
                print("[Step 9] Done — CodeFormer")
                return codeformer_rgb, info
            if method == "codeformer":
                return img_rgb, "CodeFormer not available or failed."

        # Ensure model is downloaded and loaded
        restorer = load_gfpgan()
        if restorer is None:
            print("  [GFPGAN] Model not available — returning original.")
            return img_rgb, "GFPGAN not available (model load failed). Returning original."

        # GFPGAN expects BGR input
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        _, _, restored_bgr = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        if restored_bgr is None:
            print("  [GFPGAN] Enhancement returned None.")
            return img_rgb, "GFPGAN enhancement failed."

        # Blend with original
        if blend < 1.0:
            restored_bgr = cv2.addWeighted(
                restored_bgr, blend, img_bgr, 1.0 - blend, 0
            )

        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        info = f"GFPGAN restoration complete. Blend={blend:.2f}"
        print(f"[Step 9] Done — GFPGAN")
        return restored_rgb, info

    except Exception as e:
        print(f"[Step 9] GFPGAN error: {e}")
        return img_rgb, f"Face restoration error: {e}"
