#!/usr/bin/env python3
"""
Generate vocoder calibration data on dev machine (no axengine needed).

Collects mel features from:
  1. Real mel output files (output_mel.bin from previous C++ runs)
  2. Random mel features in the expected value range
  3. Zero/silence mel features
Then runs them through ONNX to get intermediate features for head_linear calibration.

Usage:
  python3 generate_vocoder_calib.py --output-dir ./calib_data_vocoder
"""
import sys, os, argparse, json, glob
from pathlib import Path
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/ → cpp/ → repo root
ONNX_DIR = SCRIPT_DIR.parent / "vocoder_onnx"  # cpp/vocoder_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ONNX_DIR / "calib_data"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    T = 620
    mel_samples = []

    # --- Source 1: Real mel features if available (undo feat_scale to match vocoder input) ---
    feat_scale = 0.1
    for pattern in ["output_mel.bin", "output_mel_debug.bin"]:
        for p in [REPO_DIR / pattern, SCRIPT_DIR / pattern]:
            if p.exists():
                mel = np.fromfile(str(p), dtype=np.float32).reshape(-1, 100).T[np.newaxis, :, :] / feat_scale  # [1, 100, frames]
                gen_frames = min(mel.shape[2], T)
                padded = np.zeros((1, 100, T), dtype=np.float32)
                padded[0, :, :gen_frames] = mel[0, :, :gen_frames]
                mel_samples.append(padded)
                print(f"Real mel from {p}: shape={mel.shape}")

    # --- Source 2: Random mel in vocoder input range (after /feat_scale) ---
    # Python: features / 0.1 before vocoder → range ~[-8.6, 4.0]
    # So calibration mel should be in [feat_scaled] range, divided by feat_scale
    # to match the actual vocoder input distribution
    feat_scale = 0.1
    for i in range(max(0, args.num_samples - len(mel_samples))):
        length = rng.randint(100, T + 1)
        # Generate in feat_scaled range, then undo feat_scale
        mel = (rng.randn(1, 100, T).astype(np.float32) * 0.3 - 0.04) / feat_scale
        mel[0, :, length:] = 0.0
        mel_samples.append(mel)

    # --- Source 3: Edge cases (with feat_scale undo) ---
    mel_samples.append(np.zeros((1, 100, T), dtype=np.float32))
    mel_samples.append(np.ones((1, 100, T), dtype=np.float32) * 0.5 / feat_scale)
    mel_samples.append(rng.randn(1, 100, T).astype(np.float32) * 0.1 / feat_scale)
    mel_samples.append(rng.randn(1, 100, T).astype(np.float32) / feat_scale)

    print(f"\nTotal mel samples: {len(mel_samples)}")

    # --- Run ONNX to get backbone outputs (for head_linear calibration) ---
    bb_path = ONNX_DIR / "vocos_backbone_B1_T620.onnx"
    hl_path = ONNX_DIR / "vocos_head_linear_B1_T620.onnx"

    if not bb_path.exists():
        print(f"ERROR: {bb_path} not found. Run export_vocos_onnx.py first.")
        return

    sess_bb = ort.InferenceSession(str(bb_path))
    print(f"Loaded backbone: {bb_path}")

    # --- Save backbone calibration data ---
    bb_dir = out_dir / "vocos_backbone" / "mel"
    bb_dir.mkdir(parents=True, exist_ok=True)
    bb_entries = []
    backbone_outputs = []

    for i, mel in enumerate(mel_samples):
        np.save(bb_dir / f"{i:04d}.npy", mel)
        bb_entries.append({"file": f"mel/{i:04d}.npy", "shape": list(mel.shape)})

        # Run ONNX to get head_linear input
        feat = sess_bb.run(None, {'mel': mel})[0]
        backbone_outputs.append(feat)

    print(f"Backbone calibration: {len(bb_entries)} samples")

    # --- Save head_linear calibration data ---
    hl_dir = out_dir / "vocos_head_linear" / "features"
    hl_dir.mkdir(parents=True, exist_ok=True)
    hl_entries = []

    for i, feat in enumerate(backbone_outputs):
        np.save(hl_dir / f"{i:04d}.npy", feat)
        hl_entries.append({"file": f"features/{i:04d}.npy", "shape": list(feat.shape)})

    print(f"Head_linear calibration: {len(hl_entries)} samples")

    # --- Verify head_linear ONNX with calibration data ---
    if hl_path.exists():
        sess_hl = ort.InferenceSession(str(hl_path))
        for i in range(min(3, len(backbone_outputs))):
            r, im = sess_hl.run(None, {'features': backbone_outputs[i]})
            print(f"  Verify head[{i}]: real range=[{r.min():.3f},{r.max():.3f}], "
                  f"imag range=[{im.min():.3f},{im.max():.3f}]")

    # --- Manifest ---
    manifest = {
        "description": "Vocoder ONNX calibration data",
        "backbone": {
            "model": "vocos_backbone_B1_T620.onnx",
            "input": "mel", "shape": [1, 100, 620], "dtype": "float32",
            "num_samples": len(bb_entries),
            "files": bb_entries,
        },
        "head_linear": {
            "model": "vocos_head_linear_B1_T620.onnx",
            "input": "features", "shape": [1, 620, 512], "dtype": "float32",
            "num_samples": len(hl_entries),
            "files": hl_entries,
        },
    }
    with open(out_dir / "calib_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Output: {out_dir}")
    print(f"  vocos_backbone/mel/   : {len(bb_entries)} .npy files")
    print(f"  vocos_head_linear/features/: {len(hl_entries)} .npy files")


if __name__ == "__main__":
    main()
