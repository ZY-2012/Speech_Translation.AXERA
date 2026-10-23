#!/usr/bin/env python3
"""Split static ZipVoice fm_decoder ONNX into four coarse decoder parts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import onnx
from onnx import utils as onnx_utils


TIME_HIDDEN = "/fm_decoder/0/time_emb/0/Sub_3_output_0"
PADDING_MASK2 = "/Concat_1_output_0"
DISTILL_GUIDANCE_EMBED = "/fm_decoder/Concat_1_output_0"
CFG_SCALE = "/Where_1_output_0"
BLOCK1 = "/fm_decoder/1/out_combiner/Add_output_0"
BLOCK2 = "/fm_decoder/2/out_combiner/Add_output_0"
BLOCK3 = "/fm_decoder/3/out_combiner/Add_output_0"
OUTPUT = "v"

FRIENDLY_NAMES = {
    TIME_HIDDEN: "time_hidden",
    PADDING_MASK2: "padding_mask2",
    DISTILL_GUIDANCE_EMBED: "guidance_embed",
    CFG_SCALE: "cfg_scale",
    BLOCK1: "decoder_hidden_p0",
    BLOCK2: "decoder_hidden_p1",
    BLOCK3: "decoder_hidden_p2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split static ZipVoice fm_decoder ONNX into four parts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-model-dir", default=".work_tmp/full_static_distill_1024_ax630c")
    parser.add_argument("--output-dir", default="models/decoder4_static_distill_1024_ax630c")
    parser.add_argument("--work-dir", default=".work_tmp/split_decoder4_distill_1024_ax630c")
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


def _collect_node_inputs(graph: onnx.GraphProto) -> set[str]:
    used: set[str] = set()
    for node in graph.node:
        used.update(name for name in node.input if name)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                used.update(_collect_node_inputs(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    used.update(_collect_node_inputs(subgraph))
    return used


def remove_unused_graph_inputs(model_path: Path) -> None:
    model = onnx.load(str(model_path))
    used = _collect_node_inputs(model.graph)
    remove_indexes = [
        index for index, value in enumerate(model.graph.input) if value.name not in used
    ]
    if not remove_indexes:
        return
    removed = [model.graph.input[index].name for index in remove_indexes]
    for index in reversed(remove_indexes):
        del model.graph.input[index]
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))
    print(f"Removed unused input(s) from {model_path.name}: {', '.join(removed)}")


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
    input_names: List[str],
    output_names: List[str],
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
    rename_tensors(raw_path, FRIENDLY_NAMES)
    simplify_model(raw_path, final_path, enable_onnxsim=enable_onnxsim)
    remove_unused_graph_inputs(final_path)
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
    source_model_dir = Path(args.source_model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    source_decoder = source_model_dir / "fm_decoder.onnx"
    source_encoder = source_model_dir / "encoder_core.onnx"
    source_runtime_config = source_model_dir / "runtime_config.json"

    if not source_decoder.is_file():
        raise FileNotFoundError(source_decoder)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", str(work_dir / "tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(work_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(work_dir / "xdg_cache"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    inferred_source = work_dir / "fm_decoder.shape_inferred.onnx"
    inferred = onnx.shape_inference.infer_shapes(onnx.load(str(source_decoder)))
    onnx.save(inferred, str(inferred_source))
    available_names = collect_available_names(inferred)

    has_cfg_scale = CFG_SCALE in available_names
    if PADDING_MASK2 in available_names:
        mask_name = PADDING_MASK2
    else:
        # ZipVoice-Distill does not duplicate the batch for CFG.  Later
        # decoder blocks consume the original graph input mask directly.
        mask_name = "padding_mask"

    enable_onnxsim = not args.no_onnxsim
    manifest: Dict[str, object] = {
        "version": 1,
        "source_model_dir": str(source_model_dir),
        "source_fm_decoder": str(source_decoder),
        "split_type": "decoder4_block1_block2_block3",
        "friendly_names": FRIENDLY_NAMES,
        "models": [],
    }

    manifest["models"].append(
        extract_piece(
            inferred_source,
            output_dir,
            work_dir,
            "fm_decoder_part0",
            ["t", "x", "text_condition", "speech_condition", "guidance_scale", "padding_mask"],
            [BLOCK1, TIME_HIDDEN, *([PADDING_MASK2] if mask_name == PADDING_MASK2 else []), *([CFG_SCALE] if has_cfg_scale else [])],
            enable_onnxsim,
            args.keep_intermediate,
        )
    )
    manifest["models"].append(
        extract_piece(
            inferred_source,
            output_dir,
            work_dir,
            "fm_decoder_part1",
            [BLOCK1, TIME_HIDDEN, mask_name],
            [BLOCK2],
            enable_onnxsim,
            args.keep_intermediate,
        )
    )
    manifest["models"].append(
        extract_piece(
            inferred_source,
            output_dir,
            work_dir,
            "fm_decoder_part2",
            [BLOCK2, TIME_HIDDEN, mask_name],
            [BLOCK3],
            enable_onnxsim,
            args.keep_intermediate,
        )
    )
    manifest["models"].append(
        extract_piece(
            inferred_source,
            output_dir,
            work_dir,
            "fm_decoder_part3",
            [BLOCK3, TIME_HIDDEN, mask_name, *([CFG_SCALE] if has_cfg_scale else [])],
            [OUTPUT],
            enable_onnxsim,
            args.keep_intermediate,
        )
    )

    if source_encoder.is_file():
        shutil.copy2(source_encoder, output_dir / "encoder_core.onnx")
    if source_runtime_config.is_file():
        shutil.copy2(source_runtime_config, output_dir / "runtime_config.json")

    with open(output_dir / "decoder4_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if not args.keep_intermediate and inferred_source.exists():
        inferred_source.unlink()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
