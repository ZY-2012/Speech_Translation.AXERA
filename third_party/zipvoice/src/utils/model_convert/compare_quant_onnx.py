#!/usr/bin/env python3
"""Compare original ONNX and Pulsar2 quantized ONNX on prepared cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


PARTS = {
    "encoder_core": "build-encoder_nolog-check0-distill-rich-u16-fp32-ax630c",
    "fm_decoder_part0": "build-fm_decoder_part0-check0-distill-rich-u16-fp32-ax630c",
    "fm_decoder_part1": "build-fm_decoder_part1-check0-distill-rich-u16-fp32-nosm-ax630c",
    "fm_decoder_part2": "build-fm_decoder_part2-check0-distill-rich-u16-fp32-ax630c",
    "fm_decoder_part3": "build-fm_decoder_part3-check0-distill-rich-u16-fp32-subgraph-ax630c",
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="model_convert/axmodel_compare_distill_rich")
    parser.add_argument("--build-root", default="model_convert")
    args = parser.parse_args()
    cases = Path(args.cases)
    build_root = Path(args.build_root)
    manifest = __import__("json").loads((cases / "manifest.json").read_text())
    for part, build_name in PARTS.items():
        model = build_root / build_name / "quant/quant_axmodel.onnx"
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        info = manifest["parts"][part]
        for case in sorted((cases / part).iterdir()):
            if not case.is_dir():
                continue
            feed = {name: np.load(case / f"input_{name}.npy") for name in info["inputs"]}
            if part == "encoder_core":
                feed["cat_tokens"] = feed["cat_tokens"].astype(np.int64)
            outputs = session.run(None, feed)
            for name, value in zip(info["outputs"], outputs):
                ref = np.load(case / f"ref_{name}.npy")
                print(f"{part:18s} {case.name} {name:20s} cosine={cosine(ref, value):.6f} mse={np.mean((ref.astype(np.float64)-value.astype(np.float64))**2):.6g}")


if __name__ == "__main__":
    main()
