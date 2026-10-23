#!/usr/bin/env python3
"""Run AXModels on prepared cases and compare each output with ONNX references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.zipvoice_runtime import AxeSession
except ModuleNotFoundError:
    # Direct execution from scripts/ puts that directory on sys.path.
    from zipvoice_runtime import AxeSession


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="Directory made by prepare_axmodel_compare.py")
    parser.add_argument("--models", required=True, help="Directory containing encoder.axmodel and decoder_part*.axmodel")
    parser.add_argument("--output", default="axmodel_compare_results.json")
    args = parser.parse_args()

    cases_dir = Path(args.cases)
    models_dir = Path(args.models)
    manifest = json.loads((cases_dir / "manifest.json").read_text(encoding="utf-8"))
    model_files = {
        "encoder_core": "encoder.axmodel",
        "fm_decoder_part0": "decoder_part0.axmodel",
        "fm_decoder_part1": "decoder_part1.axmodel",
        "fm_decoder_part2": "decoder_part2.axmodel",
        "fm_decoder_part3": "decoder_part3.axmodel",
    }
    results = []
    for part, info in manifest["parts"].items():
        session = AxeSession(models_dir / model_files[part])
        for case_dir in sorted((cases_dir / part).iterdir()):
            if not case_dir.is_dir():
                continue
            feed = {}
            for name in info["inputs"]:
                value = np.load(case_dir / f"input_{name}.npy")
                if name == "cat_tokens":
                    value = value.astype(np.int32)
                elif name == "padding_mask":
                    value = value.astype(np.uint8)
                feed[name] = value
            actual = session.run(feed)
            for output_name in info["outputs"]:
                ref = np.load(case_dir / f"ref_{output_name}.npy")
                value = actual.get(output_name)
                if value is None:
                    value = actual[session.output_names[info["outputs"].index(output_name)]]
                results.append({
                    "part": part,
                    "case": case_dir.name,
                    "output": output_name,
                    "shape_ref": list(ref.shape),
                    "shape_actual": list(value.shape),
                    "cosine": cosine(ref, value),
                    "mse": float(np.mean((ref.astype(np.float64) - value.astype(np.float64)) ** 2)),
                    "max_abs": float(np.max(np.abs(ref.astype(np.float64) - value.astype(np.float64)))),
                })

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for item in results:
        print(f"{item['part']:18s} {item['case']} {item['output']:20s} cosine={item['cosine']:.6f} mse={item['mse']:.6g}")


if __name__ == "__main__":
    main()
