#!/usr/bin/env python3
"""Generate calibration data for the full Vocos ONNX model."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT_DIR / "model_convert" / "vocoder_onnx"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ONNX_DIR / "calib_data"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    full_path = ONNX_DIR / "vocos_full_B1_T620.onnx"
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} not found. Run scripts/vocoder/01_export_onnx.sh first.")

    frames = 620
    mel_samples = []

    # Real mel features if available. Runtime divides features by feat_scale before vocoder.
    feat_scale = 0.1
    for pattern in ["output_mel.bin", "output_mel_debug.bin"]:
        path = ROOT_DIR / pattern
        if path.exists():
            mel = (
                np.fromfile(str(path), dtype=np.float32)
                .reshape(-1, 100)
                .T[np.newaxis, :, :]
                / feat_scale
            )
            valid_frames = min(mel.shape[2], frames)
            padded = np.zeros((1, 100, frames), dtype=np.float32)
            padded[0, :, :valid_frames] = mel[0, :, :valid_frames]
            mel_samples.append(padded)
            print(f"Real mel from {path}: shape={mel.shape}")

    for _ in range(max(0, args.num_samples - len(mel_samples))):
        length = rng.randint(100, frames + 1)
        mel = (rng.randn(1, 100, frames).astype(np.float32) * 0.3 - 0.04) / feat_scale
        mel[0, :, length:] = 0.0
        mel_samples.append(mel)

    # Edge cases.
    mel_samples.append(np.zeros((1, 100, frames), dtype=np.float32))
    mel_samples.append(np.ones((1, 100, frames), dtype=np.float32) * 0.5 / feat_scale)
    mel_samples.append(rng.randn(1, 100, frames).astype(np.float32) * 0.1 / feat_scale)
    mel_samples.append(rng.randn(1, 100, frames).astype(np.float32) / feat_scale)

    print(f"\nTotal mel samples: {len(mel_samples)}")

    mel_dir = out_dir / "vocos_full" / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for i, mel in enumerate(mel_samples):
        np.save(mel_dir / f"{i:04d}.npy", mel)
        entries.append({"file": f"mel/{i:04d}.npy", "shape": list(mel.shape)})

    manifest = {
        "description": "Full Vocos ONNX calibration data",
        "full": {
            "model": "vocos_full_B1_T620.onnx",
            "input": "mel",
            "shape": [1, 100, 620],
            "dtype": "float32",
            "num_samples": len(entries),
            "files": entries,
        },
    }
    with open(out_dir / "calib_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Output: {out_dir}")
    print(f"  vocos_full/mel/: {len(entries)} .npy files")


if __name__ == "__main__":
    main()
