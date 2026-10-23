#!/usr/bin/env bash
# Quantize vocoder for AX630C (check2, AX620E NPU2)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ONNX_DIR="${ROOT_DIR}/model_convert/vocoder_onnx"
CALIB_DIR="${ONNX_DIR}/calib_data"
BUILD_DIR="${ONNX_DIR}/pulsar2_build_ax630c"
AXMODEL_DIR="${ROOT_DIR}/model_convert/vocoder_ax630c"
mkdir -p "${BUILD_DIR}" "${AXMODEL_DIR}"

FULL_MODEL="${ONNX_DIR}/vocos_full_B1_T620.onnx"
FULL_CALIB_TAR="${CALIB_DIR}/vocos_full_mel.tar.gz"
CALIB_MEL_DIR="${CALIB_DIR}/vocos_full/mel"

# Build calibration tar if needed
if [ ! -f "${FULL_CALIB_TAR}" ]; then
    echo "Creating ${FULL_CALIB_TAR}..."
    cd "${CALIB_MEL_DIR}"
    tar czf "${FULL_CALIB_TAR}" *.npy
    cd "${ROOT_DIR}"
fi
echo "  Calib tar: $(ls -lh ${FULL_CALIB_TAR} | awk '{print $5}')"

if [ ! -f "${FULL_MODEL}" ]; then
    echo "ERROR: ${FULL_MODEL} not found. Run scripts/vocoder/01_export_onnx.sh first."
    exit 1
fi
if [ ! -f "${FULL_CALIB_TAR}" ]; then
    echo "ERROR: ${FULL_CALIB_TAR} not found. Run scripts/vocoder/03_generate_calib.sh first."
    exit 1
fi

TARGET_HARDWARE="AX620E"
NPU_MODE="NPU2"
CHECK_LEVEL="0"
DEFAULT_DATA_TYPE="U16"

CONFIG="${BUILD_DIR}/config_vocos_full_ax630c.json"

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
            {"tensor_name": "mel", "calibration_dataset": "${FULL_CALIB_TAR}", "calibration_format": "Numpy", "calibration_size": -1}
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

echo "=== Quantizing vocoder for AX630C (AX620E NPU2, check2) ==="
echo "Config: ${CONFIG}"

cd "${ROOT_DIR}"
pulsar2 build --config "${CONFIG}" --input_shapes "mel:1x100x620" 2>&1 | tee "${BUILD_DIR}/build_vocos_full_ax630c.log"

BUILD_OUTPUT="${BUILD_DIR}/vocos_full.axmodel"
if [ -f "${BUILD_OUTPUT}" ]; then
    cp "${BUILD_OUTPUT}" "${AXMODEL_DIR}/"
    echo ""
    echo "=== Done! ==="
    ls -lh "${AXMODEL_DIR}/vocos_full.axmodel"
else
    echo "ERROR: Build failed"
    exit 1
fi
