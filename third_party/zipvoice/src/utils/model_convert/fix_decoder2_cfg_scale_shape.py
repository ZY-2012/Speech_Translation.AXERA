#!/usr/bin/env python3
"""Fix decoder2 cfg_scale metadata to scalar shape."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set decoder2 cfg_scale input/output shape metadata to scalar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="ONNX model to patch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.model)
    model = onnx.load(str(path))
    changed = False
    for values in [model.graph.input, model.graph.output, model.graph.value_info]:
        for value in values:
            if value.name != "cfg_scale":
                continue
            dims = value.type.tensor_type.shape.dim
            if len(dims) != 0:
                del dims[:]
                changed = True
    if changed:
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        print(f"fixed cfg_scale scalar shape: {path}")
    else:
        print(f"cfg_scale already scalar or not found: {path}")


if __name__ == "__main__":
    main()
