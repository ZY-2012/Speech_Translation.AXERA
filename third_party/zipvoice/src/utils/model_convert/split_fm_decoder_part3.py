#!/usr/bin/env python3
"""Split ZipVoice fm_decoder_part3 ONNX into two smaller ONNX models.

The default split point is after decoder block /fm_decoder/4/1:

  part3a: decoder_hidden_p2, time_hidden, padding_mask -> decoder_hidden_p3a
  part3b: decoder_hidden_p3a, time_hidden, padding_mask -> v
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import onnx
from onnx import utils as onnx_utils


DEFAULT_SPLIT_TENSOR = "/fm_decoder/4/1/bypass/Add_output_0"
FRIENDLY_NAMES = {
    DEFAULT_SPLIT_TENSOR: "decoder_hidden_p3a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split fm_decoder_part3.onnx into part3a and part3b",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Source fm_decoder_part3.onnx")
    parser.add_argument("--output-dir", required=True, help="Directory for split ONNX files")
    parser.add_argument("--work-dir", required=True, help="Temporary work directory")
    parser.add_argument("--split-tensor", default=DEFAULT_SPLIT_TENSOR)
    parser.add_argument("--no-onnxsim", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args()


def shape_of(value) -> List[object]:
    dims: List[object] = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
    return dims


def summarize_io(model_path: Path) -> Dict[str, object]:
    model = onnx.load(str(model_path))
    return {
        "nodes": len(model.graph.node),
        "inputs": {value.name: shape_of(value) for value in model.graph.input},
        "outputs": {value.name: shape_of(value) for value in model.graph.output},
    }


def rename_tensors(model_path: Path, rename_map: Dict[str, str]) -> None:
    model = onnx.load(str(model_path))
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in rename_map:
                node.input[index] = rename_map[name]
        for index, name in enumerate(node.output):
            if name in rename_map:
                node.output[index] = rename_map[name]
    for values in [
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
        model.graph.initializer,
    ]:
        for value in values:
            if value.name in rename_map:
                value.name = rename_map[value.name]
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))


def simplify_model(raw_path: Path, final_path: Path, enable_onnxsim: bool) -> None:
    if not enable_onnxsim:
        shutil.copy2(raw_path, final_path)
        onnx.checker.check_model(str(final_path))
        return
    from onnxsim import simplify

    model = onnx.load(str(raw_path))
    simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim check failed for {raw_path}")
    onnx.checker.check_model(simplified)
    onnx.save(simplified, str(final_path))


def extract_piece(
    inferred_source: Path,
    output_dir: Path,
    work_dir: Path,
    name: str,
    input_names: list[str],
    output_names: list[str],
    rename_map: Dict[str, str],
    enable_onnxsim: bool,
    keep_intermediate: bool,
) -> Dict[str, object]:
    raw_path = work_dir / f"{name}.raw.onnx"
    final_path = output_dir / f"{name}.onnx"
    onnx_utils.extract_model(
        str(inferred_source),
        str(raw_path),
        input_names=input_names,
        output_names=output_names,
        check_model=True,
    )
    rename_tensors(raw_path, rename_map)
    simplify_model(raw_path, final_path, enable_onnxsim=enable_onnxsim)
    if not keep_intermediate and raw_path.exists():
        raw_path.unlink()
    info = summarize_io(final_path)
    info.update(
        {
            "name": name,
            "file": final_path.name,
            "input_names": input_names,
            "output_names": output_names,
            "onnxsim": enable_onnxsim,
        }
    )
    return info


def collect_available_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    for values in [
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
        model.graph.initializer,
    ]:
        names.update(value.name for value in values)
    for node in model.graph.node:
        names.update(name for name in node.output if name)
    return names


def main() -> None:
    args = parse_args()
    source = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", str(work_dir / "tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(work_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(work_dir / "xdg_cache"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    inferred_source = work_dir / "fm_decoder_part3.shape_inferred.onnx"
    inferred = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    available = collect_available_names(inferred)
    if args.split_tensor not in available:
        raise ValueError(f"split tensor not found: {args.split_tensor}")
    onnx.save(inferred, str(inferred_source))

    enable_onnxsim = not args.no_onnxsim
    rename_map = {args.split_tensor: "decoder_hidden_p3a"}
    manifest: Dict[str, object] = {
        "version": 1,
        "source": str(source),
        "split_tensor": args.split_tensor,
        "friendly_names": rename_map,
        "models": [],
    }
    manifest["models"].append(
        extract_piece(
            inferred_source,
            output_dir,
            work_dir,
            "fm_decoder_part3a",
            ["decoder_hidden_p2", "time_hidden", "padding_mask"],
            [args.split_tensor],
            rename_map,
            enable_onnxsim,
            args.keep_intermediate,
        )
    )
    manifest["models"].append(
        extract_piece(
            inferred_source,
            output_dir,
            work_dir,
            "fm_decoder_part3b",
            [args.split_tensor, "time_hidden", "padding_mask"],
            ["v"],
            rename_map,
            enable_onnxsim,
            args.keep_intermediate,
        )
    )
    with open(output_dir / "part3_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if not args.keep_intermediate and inferred_source.exists():
        inferred_source.unlink()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
