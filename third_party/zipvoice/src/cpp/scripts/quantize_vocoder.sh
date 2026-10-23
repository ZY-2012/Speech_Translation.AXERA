#!/usr/bin/env bash
# Quantize vocoder ONNX models to axmodel using pulsar2.
#
# Prerequisites:
#   1. export_vocos_onnx.py (generates vocoder_onnx/*.onnx)
#   2. generate_vocoder_calib.py (generates vocoder_onnx/calib_data/)
#   3. source <npu_dev_env> (activates pulsar2, e.g. $HOME/npu-codebase/script/npu_dev)
#
# Usage:
#   bash quantize_vocoder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "${PARENT_DIR}"

ONNX_DIR="${PARENT_DIR}/vocoder_onnx"
CALIB_DIR="${ONNX_DIR}/calib_data"
BUILD_DIR="${ONNX_DIR}/pulsar2_build"
AXMODEL_DIR="${ONNX_DIR}/axmodel"

# --- Config ---
TARGET_HARDWARE="${TARGET_HARDWARE:-AX650}"
NPU_MODE="${NPU_MODE:-NPU3}"
CHECK_LEVEL="${CHECK_LEVEL:-0}"
DEFAULT_DATA_TYPE="${DEFAULT_DATA_TYPE:-U16}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:--1}"

FULL_MODEL="${ONNX_DIR}/vocos_full_B1_T620.onnx"
CALIB_DIR_BB="${CALIB_DIR}/vocos_backbone"

if [ ! -f "${FULL_MODEL}" ]; then
    echo "ERROR: ${FULL_MODEL} not found. Run export_vocos_onnx.py first."
    exit 1
fi

if [ ! -d "${CALIB_DIR_BB}" ]; then
    echo "ERROR: ${CALIB_DIR_BB} not found. Run generate_vocoder_calib.py first."
    exit 1
fi

# Activate pulsar2 if not already available
if ! command -v pulsar2 >/dev/null 2>&1; then
    echo "Activating pulsar2 environment..."
    set +u
    source "${NPU_DEV_ENV:-$HOME/npu-codebase/script/npu_dev}"
    set -u
fi

mkdir -p "${BUILD_DIR}" "${AXMODEL_DIR}"

# --- Step 1: Package calibration data as tar.gz ---
echo "=== Packaging calibration data ==="

# Calibration for full model: mel -> (real, imag)
# Input tensor is "mel", calibration data in vocos_backbone/mel/
FULL_CALIB_TAR="${CALIB_DIR}/vocos_full_mel.tar.gz"
INPUT_DIR="${CALIB_DIR_BB}/mel"

if [ ! -f "${FULL_CALIB_TAR}" ]; then
    echo "Creating ${FULL_CALIB_TAR}..."
    cd "${INPUT_DIR}"
    tar czf "${FULL_CALIB_TAR}" --transform 's|.*/||' *.npy
    cd "${SCRIPT_DIR}"
fi
echo "  ${FULL_CALIB_TAR} ($(du -h "${FULL_CALIB_TAR}" | cut -f1))"

# --- Step 2: Build pulsar2 config ---
echo ""
echo "=== Quantizing vocos_full ==="

CONFIG="${BUILD_DIR}/config_vocos_full.json"

cat > "${CONFIG}" << EOF
{
    "model_type": "ONNX",
    "npu_mode": "${NPU_MODE}",
    "input": "${FULL_MODEL}",
    "output_name": "vocos_full.axmodel",
    "output_dir": "${BUILD_DIR}",
    "target_hardware": "${TARGET_HARDWARE}",
    "onnx_opt": {"disable_onnx_optimization": false, "enable_onnxsim": true},
    "quant": {
        "input_configs": [
            {"tensor_name": "mel", "calibration_dataset": "${FULL_CALIB_TAR}", "calibration_format": "Numpy", "calibration_size": ${CALIBRATION_SIZE}}
        ],
        "layer_configs": [
            {"op_type": "Pow", "data_type": "U8"},
            {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "${DEFAULT_DATA_TYPE}"}
        ],
        "calibration_method": "MinMax",
        "enable_smooth_quant": true,
        "conv_bias_data_type": "FP32",
        "precision_analysis": true,
        "precision_analysis_method": "EndToEnd",
        "disable_auto_refine_scale": true,
        "transformer_opt_level": 0
    },
    "input_processors": [{"tensor_name": "DEFAULT"}],
    "compiler": {"check": ${CHECK_LEVEL}, "enable_slice_mode": false}
}
EOF

echo "Config: ${CONFIG}"
echo "Input shapes: mel:1x100x620"

# --- Step 3: Run pulsar2 ---
echo ""
echo "=== Running pulsar2 build ==="
pulsar2 build --config "${CONFIG}" --input_shapes "mel:1x100x620" 2>&1 | tee "${BUILD_DIR}/build_vocos_full.log"

# --- Step 4: Copy output ---
BUILD_OUTPUT="${BUILD_DIR}/vocos_full.axmodel"
if [ -f "${BUILD_OUTPUT}" ]; then
    cp "${BUILD_OUTPUT}" "${AXMODEL_DIR}/"
    echo ""
    echo "=== Done! ==="
    ls -lh "${AXMODEL_DIR}/vocos_full.axmodel"
else
    echo "ERROR: Build failed, ${BUILD_OUTPUT} not found"
    exit 1
fi
