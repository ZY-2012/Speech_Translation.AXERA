#!/usr/bin/env python3
"""Run onnxsim on one ONNX file and save the simplified model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplify one ONNX model with onnxsim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-n", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path), load_external_data=False)
    from onnxsim import simplify

    simplified, ok = simplify(
        model,
        check_n=args.check_n,
        perform_optimization=True,
    )
    if not ok:
        raise RuntimeError(f"onnxsim returned ok=False for {input_path}")
    onnx.checker.check_model(simplified)
    onnx.save(simplified, str(output_path))
    print(f"onnxsim OK: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
