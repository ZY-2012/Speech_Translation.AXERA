#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(os.environ.get("OPUS_MT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pulsar2_docker import docker_pulsar2

DEFAULT_IMAGE = os.environ.get("PULSAR2_IMAGE", "")


def base_config(onnx_rel, out_name, work_name, input_shapes):
    return {
        "input": f"/workspace/export/{onnx_rel}",
        "output_dir": "/workspace/compile",
        "output_name": out_name,
        "work_dir": f"/workspace/compile/{work_name}",
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": "NPU1",
        "input_shapes": input_shapes,
        "onnx_opt": {
            "disable_onnx_optimization": False,
            "enable_onnxsim": False,
            "model_check": True,
        },
        "quant": {
            "input_configs": [],
            "calibration_method": "MinMax",
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
            "highest_mix_precision": False,
        },
        "input_processors": [],
    }


def add_input(cfg, name, calib, dtype, calib_size=64, mean=None, std=None):
    cfg["quant"]["input_configs"].append({
        "tensor_name": name,
        "calibration_dataset": f"/workspace/export/calib_data/{calib}.tar.gz",
        "calibration_format": "Numpy",
        "calibration_size": calib_size,
        "calibration_mean": mean or [],
        "calibration_std": std or [],
    })
    cfg["input_processors"].append({
        "tensor_name": name,
        "tensor_layout": "NCHW",
        "src_dtype": dtype,
        "src_layout": "NCHW",
    })


def encoder_cfg(src_len, npu_mode="NPU1", calib_ints=True):
    cfg = base_config("encoder.onnx", "encoder.axmodel", "encoder_work",
                      f"input_ids:1x{src_len},attention_mask:1x{src_len}")
    cfg["npu_mode"] = npu_mode
    if calib_ints:
        add_input(cfg, "input_ids", "input_ids", "S32")
    add_input(cfg, "attention_mask", "attention_mask", "FP32")
    return cfg


def decoder_cfg(src_len, tgt_len, npu_mode="NPU1", calib_ints=True):
    cfg = base_config("decoder.onnx", "decoder.axmodel", "decoder_work",
                      f"decoder_input_ids:1x{tgt_len},decoder_attention_mask:1x{tgt_len},"
                      f"position_ids:1x{tgt_len},encoder_hidden_states:1x{src_len}x512,"
                      f"encoder_attention_mask:1x{src_len}")
    cfg["npu_mode"] = npu_mode
    if calib_ints:
        add_input(cfg, "decoder_input_ids", "decoder_input_ids", "S32", calib_size=256)
        add_input(cfg, "decoder_attention_mask", "decoder_attention_mask", "S32", calib_size=256)
        add_input(cfg, "position_ids", "position_ids", "S32", calib_size=256)
    add_input(cfg, "encoder_hidden_states", "encoder_hidden_states", "FP32", calib_size=256)
    add_input(cfg, "encoder_attention_mask", "decoder_encoder_attention_mask", "FP32", calib_size=256)
    return cfg


def run_build(task_dir, cfg, image, log_name, timeout=3600):
    if not image:
        raise RuntimeError("set PULSAR2_IMAGE to your Pulsar2 Docker image")
    compile_dir = Path(task_dir) / "compile"
    compile_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = compile_dir / f"{log_name}_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[compile] {task_dir} {log_name} input_shapes={cfg['input_shapes']}", flush=True)
    docker_pulsar2(image, str(Path(task_dir).resolve()),
                   f"pulsar2 build --config /workspace/compile/{cfg_path.name}",
                   timeout=timeout, log_file=compile_dir / f"{log_name}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=["zh-en", "en-zh"])
    ap.add_argument("--part", default="both", choices=["encoder", "decoder", "both"])
    ap.add_argument("--src-len", type=int, default=128)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--npu-mode", default="NPU1")
    ap.add_argument("--no-int-calib", action="store_true")
    ap.add_argument("--smooth-quant", action="store_true")
    ap.add_argument("--fp32-layernorm", action="store_true")
    ap.add_argument("--transformer-opt", type=int, default=0)
    ap.add_argument("--disable-auto-refine-scale", action="store_true")
    ap.add_argument("--u16", action="store_true")
    args = ap.parse_args()
    task_dir = ROOT / args.direction
    int_calib = not args.no_int_calib

    if args.part in ("encoder", "both"):
        cfg = encoder_cfg(args.src_len, args.npu_mode, int_calib)
        if args.smooth_quant:
            cfg["quant"]["enable_smooth_quant"] = True
            cfg["quant"]["smooth_quant_threshold"] = 0.5
        if args.transformer_opt:
            cfg["quant"]["transformer_opt_level"] = args.transformer_opt
        if args.disable_auto_refine_scale:
            cfg["quant"]["disable_auto_refine_scale"] = True
        if args.u16:
            cfg["quant"]["layer_configs"] = [
                {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U16"}
            ]
        run_build(task_dir, cfg, args.image, "encoder")
    if args.part in ("decoder", "both"):
        cfg = decoder_cfg(args.src_len, args.tgt_len, args.npu_mode, int_calib)
        if args.smooth_quant:
            cfg["quant"]["enable_smooth_quant"] = True
            cfg["quant"]["smooth_quant_threshold"] = 0.5
        if args.transformer_opt:
            cfg["quant"]["transformer_opt_level"] = args.transformer_opt
        if args.disable_auto_refine_scale:
            cfg["quant"]["disable_auto_refine_scale"] = True
        if args.u16:
            cfg["quant"]["layer_configs"] = [
                {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U16"}
            ]
        if args.fp32_layernorm:
            cfg["quant"]["layer_configs"] = [
                {"op_type": "ReduceMean", "data_type": "FP32"},
                {"op_type": "Sqrt", "data_type": "FP32"},
                {"op_type": "Pow", "data_type": "FP32"},
                {"op_type": "Div", "data_type": "FP32"},
            ]
        run_build(task_dir, cfg, args.image, "decoder")


if __name__ == "__main__":
    main()
