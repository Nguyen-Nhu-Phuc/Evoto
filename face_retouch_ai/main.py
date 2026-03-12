"""
AI Face Retouch Pipeline — main orchestrator.

Runs all 9 steps in sequence:
  1. Face Detection    (RetinaFace)
  2. Face Landmarks    (MediaPipe FaceMesh 478)
  3. Face Parsing      (BiSeNet 19-class)
  4. Blemish Detection (U-Net / heuristic)
  5. Mask Expansion    (dilate + blur)
  6. AI Inpainting     (LaMa / OpenCV)
  7. Skin Smoothing    (guided-filter freq-sep)
  8. Texture Restore   (clamped high-pass)
  9. Strength Blend    (slider [0, 1])

Usage:
    python main.py --input portrait.jpg --output result.jpg --strength 0.8
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Pipeline modules
from pipelines.face_detect import detect_faces
from pipelines.landmarks import extract_landmarks
from pipelines.face_parsing import parse_face
from pipelines.blemish_seg import detect_blemishes
from pipelines.mask_expand import expand_mask
from pipelines.inpaint import inpaint
from pipelines.skin_retouch import smooth_skin
from pipelines.texture_restore import restore_texture

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEBUG_DIR = OUTPUT_DIR / "debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline(
    img_rgb: np.ndarray,
    strength: float = 0.8,
    save_debug: bool = True,
) -> dict:
    """
    Run the full 9-step retouch pipeline.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3) RGB uint8
    strength : float in [0, 1] — final retouch blending strength
    save_debug : bool — save intermediate debug images

    Returns
    -------
    dict with keys:
        result      : np.ndarray (H, W, 3) RGB uint8
        steps       : dict[str, any] — per-step outputs
        info        : str — full report
        elapsed     : float — total seconds
    """
    t0 = time.time()
    h, w = img_rgb.shape[:2]
    report = []
    steps = {}

    print("=" * 60)
    print(f"  AI Face Retouch Pipeline — {w}×{h}, strength={strength:.2f}")
    print("=" * 60)

    # Save input
    if save_debug:
        cv2.imwrite(
            str(DEBUG_DIR / "step0_input.jpg"),
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        )

    # ── Step 1: Face Detection ─────────────────────────────────────────
    faces, face_debug, face_info = detect_faces(img_rgb)
    steps["faces"] = faces
    report.append(f"[Step 1] {face_info}")
    if not faces:
        print("WARNING: No face detected — pipeline will process full image.")

    # Crop to face region (with padding) for processing
    if faces:
        bbox = faces[0]["bbox"]  # largest / first face
        pad = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.3)
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(w, bbox[2] + pad)
        y2 = min(h, bbox[3] + pad)
        face_roi = img_rgb[y1:y2, x1:x2].copy()
    else:
        face_roi = img_rgb.copy()
        x1, y1, x2, y2 = 0, 0, w, h

    # ── Step 2: Face Landmarks ─────────────────────────────────────────
    landmarks, protect_mask_full, lm_info = extract_landmarks(img_rgb)
    protect_roi = protect_mask_full[y1:y2, x1:x2]
    steps["landmarks"] = landmarks
    report.append(f"[Step 2] {lm_info}")

    # ── Step 3: Face Parsing ───────────────────────────────────────────
    parsing_map, skin_mask_full, colour_vis, parse_info = parse_face(img_rgb)
    skin_roi = skin_mask_full[y1:y2, x1:x2]
    steps["parsing_map"] = parsing_map
    steps["skin_mask"] = skin_mask_full
    report.append(f"[Step 3] {parse_info}")

    # ── Step 4: Blemish Detection ──────────────────────────────────────
    blemish_mask_roi, blemish_method, blem_info = detect_blemishes(
        face_roi, skin_roi, protect_roi
    )
    steps["blemish_mask"] = blemish_mask_roi
    steps["blemish_method"] = blemish_method
    report.append(f"[Step 4] {blem_info}")

    # ── Step 5: Mask Expansion ─────────────────────────────────────────
    soft_mask_roi, mask_info = expand_mask(blemish_mask_roi)
    steps["soft_mask"] = soft_mask_roi
    report.append(f"[Step 5] {mask_info}")

    # ── Step 6: AI Inpainting ──────────────────────────────────────────
    inpainted_roi, inp_method, inp_info = inpaint(
        face_roi, blemish_mask_roi, soft_mask_roi
    )
    steps["inpaint_method"] = inp_method
    report.append(f"[Step 6] {inp_info}")

    # ── Step 7: Skin Smoothing ─────────────────────────────────────────
    smoothed_roi, smooth_info = smooth_skin(inpainted_roi, skin_roi)
    report.append(f"[Step 7] {smooth_info}")

    # ── Step 8: Texture Restore ────────────────────────────────────────
    restored_roi, tex_info = restore_texture(smoothed_roi, skin_roi)
    report.append(f"[Step 8] {tex_info}")

    # ── Step 9: Strength Blend ─────────────────────────────────────────
    print(f"[Step 9] Strength Blend — {strength:.2f}")
    blended_roi = (
        face_roi.astype(np.float32) * (1 - strength)
        + restored_roi.astype(np.float32) * strength
    ).clip(0, 255).astype(np.uint8)

    # Paste back into full image
    result = img_rgb.copy()
    result[y1:y2, x1:x2] = blended_roi

    # Save final result
    if save_debug:
        cv2.imwrite(
            str(DEBUG_DIR / "step9_blended_roi.jpg"),
            cv2.cvtColor(blended_roi, cv2.COLOR_RGB2BGR),
        )

    elapsed = time.time() - t0

    # Final diff stats
    diff = np.abs(result.astype(float) - img_rgb.astype(float))
    pix_changed = np.any(diff > 10, axis=2).sum()
    final_info = (
        f"\n{'='*60}\n"
        f"  FINAL RESULTS\n"
        f"  Mean diff:       {diff.mean():.2f}/255\n"
        f"  Max diff:        {diff.max():.0f}\n"
        f"  Pixels Δ>10:     {pix_changed} ({pix_changed/(h*w)*100:.1f}%)\n"
        f"  Strength:        {strength:.2f}\n"
        f"  Blemish method:  {blemish_method}\n"
        f"  Inpaint method:  {inp_method}\n"
        f"  Elapsed:         {elapsed:.1f}s\n"
        f"{'='*60}"
    )
    report.append(final_info)
    print(final_info)

    return {
        "result": result,
        "steps": steps,
        "info": "\n".join(report),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="AI Face Retouch Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    parser.add_argument(
        "--strength", "-s", type=float, default=0.8, help="Retouch strength [0-1]"
    )
    args = parser.parse_args()

    inp_path = Path(args.input)
    if not inp_path.exists():
        raise FileNotFoundError(f"Input image not found: {inp_path}")

    # Read image (BGR → RGB)
    bgr = cv2.imread(str(inp_path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {inp_path}")
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"Input: {inp_path}  ({img_rgb.shape[1]}×{img_rgb.shape[0]})")

    out = run_pipeline(img_rgb, strength=args.strength)

    # Save output
    out_path = args.output or str(OUTPUT_DIR / f"{inp_path.stem}_retouched.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(out["result"], cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
