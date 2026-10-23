#!/usr/bin/env python3
"""Create AX630C check0 configs for targeted decoder mixed precision tuning.

This generates only configs. It does not run Pulsar2.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONVERT = ROOT / "model_convert"
ONNX_DIR = ROOT / "models" / "decoder4_static_distill_1024_ax630c"


PARTS = {
    "part0": {
        "base_config": MODEL_CONVERT
        / "config_fm_decoder_part0_check2-distill-rich-u16-fp32-ax630c-reexport.json",
        "output_config": MODEL_CONVERT
        / "config_fm_decoder_part0_check0-distill-rich-u16-fp32-mosaicfix-ax630c-reexport.json",
        "output_dir": "./build-fm_decoder_part0-check0-distill-rich-u16-fp32-mosaicfix-ax630c-reexport",
        "onnx": ONNX_DIR / "fm_decoder_part0.onnx",
        "hardsoftplus": [
            "HardSoftplus_5",
            "HardSoftplus_7",
            "HardSoftplus_9",
            "HardSoftplus_10",
            "HardSoftplus_12",
            "HardSoftplus_14",
            "HardSoftplus_15",
            "HardSoftplus_17",
            "HardSoftplus_19",
        ],
        "extra_layer_substrings": [],
        "extra_layer_names": [],
    },
    "part2": {
        "base_config": MODEL_CONVERT
        / "config_fm_decoder_part2_check2-distill-rich-u16-fp32-ax630c-reexport.json",
        "output_config": MODEL_CONVERT
        / "config_fm_decoder_part2_check0-distill-rich-u16-fp32-mosaicfix-ax630c-reexport.json",
        "output_dir": "./build-fm_decoder_part2-check0-distill-rich-u16-fp32-mosaicfix-ax630c-reexport",
        "onnx": ONNX_DIR / "fm_decoder_part2.onnx",
        "hardsoftplus": [
            "HardSoftplus_45",
            "HardSoftplus_47",
            "HardSoftplus_49",
            "HardSoftplus_50",
            "HardSoftplus_52",
            "HardSoftplus_54",
            "HardSoftplus_55",
            "HardSoftplus_57",
            "HardSoftplus_59",
        ],
        "extra_layer_substrings": [],
        "extra_layer_names": ["/fm_decoder/3/encoder/0/norm/Pow"],
    },
    "part3": {
        "base_config": MODEL_CONVERT
        / "config_fm_decoder_part3_check2-distill-rich-u16-fp32-subgraph-ax630c-reexport.json",
        "output_config": MODEL_CONVERT
        / "config_fm_decoder_part3_check0-distill-rich-u16-fp32-mosaicfix-ax630c-reexport.json",
        "output_dir": "./build-fm_decoder_part3-check0-distill-rich-u16-fp32-mosaicfix-ax630c-reexport",
        "onnx": ONNX_DIR / "fm_decoder_part3.onnx",
        "hardsoftplus": [
            "HardSoftplus_70",
            "HardSoftplus_72",
            "HardSoftplus_74",
            "HardSoftplus_75",
            "HardSoftplus_77",
            "HardSoftplus_79",
        ],
        "extra_layer_substrings": [
            "/fm_decoder/4/3/self_attn2/",
            "/fm_decoder/4/3/conv_module2/",
        ],
        "extra_layer_names": [],
    },
}


def collect_layer_names(onnx_path: Path, substrings: list[str]) -> list[str]:
    model = onnx.load(str(onnx_path))
    names = []
    for node in model.graph.node:
        if node.name and any(s in node.name for s in substrings):
            names.append(node.name)
    return sorted(set(names))


def main() -> None:
    for part, spec in PARTS.items():
        cfg = json.loads(spec["base_config"].read_text())
        cfg["output_dir"] = spec["output_dir"]
        cfg["compiler"]["check"] = 0

        layer_configs = copy.deepcopy(cfg["quant"]["layer_configs"])

        hardsoftplus_layers = collect_layer_names(spec["onnx"], spec["hardsoftplus"])
        if hardsoftplus_layers:
            layer_configs.append(
                {
                    "layer_names": hardsoftplus_layers,
                    "data_type": "FP32",
                }
            )

        extra_layers = collect_layer_names(spec["onnx"], spec["extra_layer_substrings"])
        extra_layers.extend(spec["extra_layer_names"])
        extra_layers = sorted(set(extra_layers))

        # Pulsar2 validates layer names after frontend optimization. Some MatMul/Add
        # pairs become Gemm and no longer exist by their original ONNX names.
        optimized_onnx = MODEL_CONVERT / spec["output_dir"].lstrip("./") / "frontend" / "optimized.onnx"
        if optimized_onnx.exists():
            optimized_names = {node.name for node in onnx.load(str(optimized_onnx)).graph.node}
            before_hs = len(hardsoftplus_layers)
            before_extra = len(extra_layers)
            hardsoftplus_layers = [name for name in hardsoftplus_layers if name in optimized_names]
            extra_layers = [name for name in extra_layers if name in optimized_names]
            print(
                f"{part}: filtered by optimized.onnx "
                f"hardsoftplus {before_hs}->{len(hardsoftplus_layers)} "
                f"extra {before_extra}->{len(extra_layers)}"
            )

        if extra_layers:
            layer_configs.append(
                {
                    "layer_names": extra_layers,
                    "data_type": "FP32",
                }
            )

        cfg["quant"]["layer_configs"] = layer_configs
        spec["output_config"].write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            f"{part}: wrote {spec['output_config']} "
            f"hardsoftplus_layers={len(hardsoftplus_layers)} "
            f"extra_layers={len(extra_layers)}"
        )


if __name__ == "__main__":
    main()
