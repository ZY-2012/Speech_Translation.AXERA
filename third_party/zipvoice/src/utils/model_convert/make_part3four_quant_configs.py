#!/usr/bin/env python3
"""Create Pulsar2 check0 configs for four-way split fm_decoder_part3."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/data/shared/huyuan/TTS_quant/ZipVoice.AXERA")
MODEL_DIR = ROOT / "models" / "decoder4_static_distill_1024_ax630c_part3four"
CALIB_DIR = ROOT / "model_convert" / "calib_data_decoder4_rich_distill_1024_part3four"
OUT_DIR = ROOT / "model_convert"

SPECS = [
    ("fm_decoder_part3_0", "decoder_part3_0.axmodel", "decoder_hidden_p2", "decoder_hidden_p30"),
    ("fm_decoder_part3_1", "decoder_part3_1.axmodel", "decoder_hidden_p30", "decoder_hidden_p31"),
    ("fm_decoder_part3_2", "decoder_part3_2.axmodel", "decoder_hidden_p31", "decoder_hidden_p32"),
    ("fm_decoder_part3_3", "decoder_part3_3.axmodel", "decoder_hidden_p32", "v"),
]


def main() -> None:
    for model_name, axmodel_name, hidden_input, _output in SPECS:
        cfg = {
            "model_type": "ONNX",
            "npu_mode": "NPU2",
            "input": str(MODEL_DIR / f"{model_name}.onnx"),
            "output_name": axmodel_name,
            "output_dir": f"./build-{model_name}-check0-distill-rich-four-u16-powu8-ax630c",
            "target_hardware": "AX620E",
            "onnx_opt": {
                "disable_onnx_optimization": False,
                "enable_onnxsim": True,
            },
            "quant": {
                "input_configs": [
                    {
                        "tensor_name": hidden_input,
                        "calibration_dataset": str(
                            CALIB_DIR / model_name / f"{hidden_input}.tar.gz"
                        ),
                        "calibration_format": "Numpy",
                        "calibration_size": -1,
                    },
                    {
                        "tensor_name": "time_hidden",
                        "calibration_dataset": str(
                            CALIB_DIR / model_name / "time_hidden.tar.gz"
                        ),
                        "calibration_format": "Numpy",
                        "calibration_size": -1,
                    },
                    {
                        "tensor_name": "padding_mask",
                        "calibration_dataset": str(
                            CALIB_DIR / model_name / "padding_mask.tar.gz"
                        ),
                        "calibration_format": "Numpy",
                        "calibration_size": -1,
                    },
                ],
                "layer_configs": [
                    {"op_type": "Pow", "data_type": "U8"},
                    {
                        "start_tensor_names": ["DEFAULT"],
                        "end_tensor_names": ["DEFAULT"],
                        "data_type": "U16",
                    },
                ],
                "calibration_method": "MinMax",
                "enable_smooth_quant": True,
                "conv_bias_data_type": "FP32",
                "precision_analysis": True,
                "precision_analysis_method": "EndToEnd",
                "disable_auto_refine_scale": True,
                "transformer_opt_level": 0,
            },
            "input_processors": [{"tensor_name": "DEFAULT"}],
            "compiler": {"check": 0, "enable_slice_mode": False},
        }
        path = OUT_DIR / f"config_{model_name}_check0-distill-rich-four-u16-powu8-ax630c.json"
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
