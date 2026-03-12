"""
download_models.py
==================
Automatically download pretrained model checkpoints required by the
AI portrait blemish-removal pipeline.  Skips any file that already exists.

Usage:
    python utils/download_models.py
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile


# ---------------------------------------------------------------------------
# Configuration – model name → download URL → local filename
# ---------------------------------------------------------------------------
# NOTE: Some research models are distributed via Google Drive / GitHub
# releases.  The URLs below point to commonly-hosted public checkpoints.
# If a URL becomes stale, replace it with an up-to-date mirror.
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_REGISTRY = {
    # RetinaFace (InsightFace buffalo_l pack – contains det_10g.onnx)
    "retinaface": {
        "url": (
            "https://github.com/deepinsight/insightface/releases/"
            "download/v0.7/buffalo_l.zip"
        ),
        "filename": "buffalo_l.zip",
        "description": "RetinaFace detector from InsightFace (buffalo_l)",
        "gdrive_id": None,
        "unzip": True,
    },
    # BiSeNet face-parsing (hosted on Google Drive by the author)
    "face_parsing": {
        "url": None,
        "gdrive_id": "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
        "filename": "face_parsing_79999_iter.pth",
        "description": "BiSeNet face-parsing pretrained weights",
        "unzip": False,
    },
    # LaMa inpainting (big-lama checkpoint – Hugging Face)
    "lama": {
        "url": "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip",
        "gdrive_id": None,
        "filename": "big-lama.zip",
        "description": "LaMa inpainting model (big-lama)",
        "unzip": True,
    },
    # CodeFormer face restoration
    "codeformer": {
        "url": (
            "https://github.com/sczhou/CodeFormer/releases/"
            "download/v0.1.0/codeformer.pth"
        ),
        "filename": "codeformer.pth",
        "description": "CodeFormer face-restoration weights",
        "gdrive_id": None,
        "unzip": False,
    },
    # RestoreFormer – face restoration (VQGAN-based)
    "restoreformer": {
        "url": (
            "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/"
            "download/v1.0.0/RestoreFormer.ckpt"
        ),
        "gdrive_id": None,
        "filename": "RestoreFormer.ckpt",
        "description": "RestoreFormer face-restoration weights",
        "unzip": False,
    },
}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, description: str = "") -> bool:
    """
    Download a file from *url* to *dest* with a progress bar.

    Returns True on success, False on failure.
    """
    try:
        # Validate inputs
        if not url or not isinstance(url, str):
            print(f"[ERROR] Invalid URL provided for {description}.")
            return False
        if not dest or not isinstance(dest, Path):
            print(f"[ERROR] Invalid destination path for {description}.")
            return False

        print(f"\n>> Downloading: {description}")
        print(f"   URL : {url}")
        print(f"   Dest: {dest}")

        # Stream the download so large files don't blow up memory
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as fh, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))

        print(f"   [OK] Saved to {dest}")
        return True

    except requests.exceptions.RequestException as exc:
        print(f"   [ERROR] Network error downloading {description}: {exc}")
        return False
    except OSError as exc:
        print(f"   [ERROR] File-system error saving {description}: {exc}")
        return False
    except Exception as exc:
        print(f"   [ERROR] Unexpected error downloading {description}: {exc}")
        return False


def download_from_gdrive(gdrive_id: str, dest: Path, description: str = "") -> bool:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown

        print(f"\n>> Downloading (Google Drive): {description}")
        print(f"   File ID: {gdrive_id}")
        print(f"   Dest   : {dest}")

        url = f"https://drive.google.com/uc?id={gdrive_id}"
        output = gdown.download(url, str(dest), quiet=False)
        if output and Path(output).exists():
            print(f"   [OK] Saved to {dest}")
            return True
        else:
            print(f"   [ERROR] gdown returned no output for {description}")
            return False
    except ImportError:
        print("   [ERROR] gdown is not installed. Run: pip install gdown")
        return False
    except Exception as exc:
        print(f"   [ERROR] Google Drive download failed for {description}: {exc}")
        return False


def unzip_file(zip_path: Path, extract_to: Path) -> bool:
    """Extract a ZIP archive into *extract_to* directory."""
    try:
        print(f"   Extracting {zip_path.name} → {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        print(f"   [OK] Extracted successfully")
        return True
    except Exception as exc:
        print(f"   [ERROR] Extraction failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def download_all_models() -> None:
    """Download every registered model checkpoint, skipping those already on disk."""
    try:
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Models directory: {MODELS_DIR}\n")

        success_count = 0
        skip_count = 0
        fail_count = 0

        for name, info in MODEL_REGISTRY.items():
            dest = MODELS_DIR / info["filename"]
            gdrive_id = info.get("gdrive_id")
            should_unzip = info.get("unzip", False)
            url = info.get("url")

            # Skip if file already exists (or extracted folder exists for zips)
            if dest.exists():
                print(f"[SKIP] {info['description']} already exists at {dest}")
                skip_count += 1
                continue

            # For zips, also check if the extracted folder already exists
            if should_unzip and dest.suffix == ".zip":
                extracted_dir = MODELS_DIR / dest.stem
                if extracted_dir.exists() and extracted_dir.is_dir():
                    print(f"[SKIP] {info['description']} already extracted at {extracted_dir}")
                    skip_count += 1
                    continue

            # Choose download method
            if gdrive_id:
                ok = download_from_gdrive(gdrive_id, dest, info["description"])
            elif url:
                ok = download_file(url, dest, info["description"])
            else:
                print(f"[SKIP] {info['description']} — no download URL available")
                skip_count += 1
                continue

            if ok:
                success_count += 1
                # Auto-extract if flagged
                if should_unzip and dest.suffix == ".zip":
                    unzip_file(dest, MODELS_DIR)
            else:
                fail_count += 1

        # Summary
        print("\n" + "=" * 60)
        print(f"Download complete:  {success_count} downloaded, "
              f"{skip_count} skipped, {fail_count} failed.")
        print("=" * 60)

    except Exception as exc:
        print(f"[FATAL] Could not complete model downloads: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    download_all_models()
