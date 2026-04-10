"""
download_models.py
==================
Audit and download required models for face_retouch_ai.

Usage:
  python utils/download_models.py                 # audit + download auto models
  python utils/download_models.py --check-only    # only show current status
"""

import argparse
import shutil
import sys
from pathlib import Path
import zipfile

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


MODEL_SPECS = [
    {
        "name": "GFPGAN",
        "required_path": MODELS_DIR / "face_restore" / "GFPGANv1.4.pth",
        "description": "Face restoration fallback",
        "url": (
            "https://github.com/TencentARC/GFPGAN/releases/"
            "download/v1.3.0/GFPGANv1.4.pth"
        ),
    },
    {
        "name": "MediaPipe Face Landmarker",
        "required_path": MODELS_DIR / "mediapipe" / "face_landmarker.task",
        "description": "478-point landmark model",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
    },
    {
        "name": "Face Parsing BiSeNet",
        "required_path": MODELS_DIR / "face_parsing" / "79999_iter.pth",
        "description": "Face parsing segmentation weights",
        "gdrive_id": "154JgKpzCPW82qINcVieuPH3fZ2e0P812",
    },
    {
        "name": "InsightFace buffalo_l",
        "required_path": MODELS_DIR / "insightface" / "models" / "buffalo_l",
        "description": "RetinaFace detector pack",
        "url": (
            "https://github.com/deepinsight/insightface/releases/"
            "download/v0.7/buffalo_l.zip"
        ),
        "archive_temp": MODELS_DIR / "_tmp" / "buffalo_l.zip",
        "extract_to": MODELS_DIR / "_tmp" / "buffalo_l",
        "post_move": {
            "src": "buffalo_l",
            "dst": MODELS_DIR / "insightface" / "models" / "buffalo_l",
        },
    },
    {
        "name": "CodeFormer runtime repo",
        "required_path": MODELS_DIR / "face_restore" / "codeformer" / "inference_codeformer.py",
        "description": "Optional, used by method=codeformer",
        "manual_note": (
            "Clone repo into models/face_restore/codeformer:\n"
            "  git clone https://github.com/sczhou/CodeFormer "
            "models/face_restore/codeformer\n"
            "Then install its requirements in your environment."
        ),
    },
]


def _download_file(url: str, dest: Path, label: str) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n>> Downloading {label}")
        print(f"   URL : {url}")
        print(f"   Dest: {dest}")
        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"   [ERROR] {label}: {exc}")
        return False


def _download_gdrive(gdrive_id: str, dest: Path, label: str) -> bool:
    try:
        import gdown
    except ImportError:
        print("   [ERROR] gdown is not installed. Run: pip install gdown")
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        print(f"\n>> Downloading {label} (Google Drive)")
        print(f"   Dest: {dest}")
        out = gdown.download(url, str(dest), quiet=False)
        return bool(out and dest.exists())
    except Exception as exc:
        print(f"   [ERROR] {label}: {exc}")
        return False


def _extract_zip(zip_path: Path, extract_to: Path) -> bool:
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    except Exception as exc:
        print(f"   [ERROR] Extract failed: {exc}")
        return False


def _install_insightface_from_zip(spec: dict) -> bool:
    archive_path = spec["archive_temp"]
    extract_to = spec["extract_to"]
    move_cfg = spec["post_move"]
    dst = move_cfg["dst"]
    src_name = move_cfg["src"]
    if not _download_file(spec["url"], archive_path, spec["name"]):
        return False
    if not _extract_zip(archive_path, extract_to):
        return False
    src = extract_to / src_name
    if not src.exists():
        # Some archives are flat (onnx files directly under extract_to).
        onnx_files = list(extract_to.glob("*.onnx"))
        if onnx_files:
            src = extract_to
        else:
            print(f"   [ERROR] Expected extracted folder missing: {src}")
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(src), str(dst))
    print(f"   [OK] Installed to {dst}")
    return True


def audit_models() -> list[dict]:
    rows = []
    for spec in MODEL_SPECS:
        exists = spec["required_path"].exists()
        rows.append(
            {
                "name": spec["name"],
                "path": spec["required_path"],
                "exists": exists,
                "description": spec["description"],
                "can_auto_download": bool(spec.get("url") or spec.get("gdrive_id")),
                "manual_note": spec.get("manual_note"),
            }
        )
    return rows


def print_audit(rows: list[dict]) -> None:
    print(f"Project models dir: {MODELS_DIR}\n")
    for row in rows:
        status = "OK" if row["exists"] else "MISSING"
        print(f"[{status}] {row['name']}")
        print(f"  - Path: {row['path']}")
        print(f"  - Use : {row['description']}")
        if not row["exists"] and row["manual_note"]:
            print("  - Note:")
            for line in row["manual_note"].splitlines():
                print(f"    {line}")
    ok_count = sum(1 for r in rows if r["exists"])
    print(f"\nSummary: {ok_count}/{len(rows)} required model entries available.")


def download_missing(rows: list[dict]) -> None:
    downloaded = 0
    failed = 0
    skipped = 0
    for spec in MODEL_SPECS:
        path = spec["required_path"]
        if path.exists():
            skipped += 1
            continue
        if spec["name"] == "InsightFace buffalo_l":
            ok = _install_insightface_from_zip(spec)
        elif spec.get("gdrive_id"):
            ok = _download_gdrive(spec["gdrive_id"], path, spec["name"])
        elif spec.get("url"):
            ok = _download_file(spec["url"], path, spec["name"])
        else:
            ok = False
            print(f"\n>> {spec['name']} requires manual install.")
            note = spec.get("manual_note", "No manual note provided.")
            for line in note.splitlines():
                print(f"   {line}")
        if ok:
            downloaded += 1
        else:
            failed += 1
    print(
        "\nDownload summary: "
        f"{downloaded} downloaded, {skipped} already present, {failed} failed/manual."
    )


def download_modelscope_skin_retouching() -> bool:
    """Download iic/cv_unet_skin-retouching into models/modelscope/ (ModelScope SDK)."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("   [ERROR] modelscope is not installed. Run: pip install modelscope")
        return False
    dest = MODELS_DIR / "modelscope" / "iic_cv_unet_skin_retouching"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        print(f"\n>> ModelScope skin retouch: {dest}")
        snapshot_download("iic/cv_unet_skin-retouching", local_dir=str(dest))
        return True
    except Exception as exc:
        print(f"   [ERROR] ModelScope skin download: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit/download face_retouch_ai models")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only audit models without downloading.",
    )
    parser.add_argument(
        "--modelscope-skin",
        action="store_true",
        help="Download iic/cv_unet_skin-retouching (TensorFlow + PyTorch weights; face detector loads on first pipeline use).",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.modelscope_skin:
        download_modelscope_skin_retouching()
    rows = audit_models()
    print_audit(rows)
    if args.check_only:
        return
    print("\nStarting auto-download for missing supported models...")
    download_missing(rows)
    print("\nRe-check after download:")
    print_audit(audit_models())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
