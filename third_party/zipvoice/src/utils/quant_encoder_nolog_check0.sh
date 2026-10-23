#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/model_convert"

WORK_TMP="${WORK_TMP:-${ROOT}/.work_tmp}"
TMPDIR="${TMPDIR:-${WORK_TMP}/tmp}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_TMP}/matplotlib}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_TMP}/xdg_cache}"
export TMPDIR MPLCONFIGDIR XDG_CACHE_HOME
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if ! command -v pulsar2 >/dev/null 2>&1; then
    set +u
    source "${NPU_DEV_ENV:-/data/huyuan/npu-codebase/script/npu_dev}"
    set -u
fi

MODEL_DIR="${MODEL_DIR:-${ROOT}/models/decoder4_static}"
CALIB_DIR="${CALIB_DIR:-${ROOT}/model_convert/calib_data_decoder4_rich}"
AXMODEL_DIR="${AXMODEL_DIR:-${ROOT}/model_convert/axmodels_encoder_nolog_rich_check0}"
MAX_TOKENS="${MAX_TOKENS:-384}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:--1}"
CHECK_LEVEL="${CHECK_LEVEL:-0}"
DEFAULT_DATA_TYPE="${DEFAULT_DATA_TYPE:-U16}"
BUILD_SUFFIX="${BUILD_SUFFIX:-}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
OUTPUT_NAME="${OUTPUT_NAME:-encoder_nolog_check0${OUTPUT_SUFFIX}.axmodel}"
TARGET_HARDWARE="${TARGET_HARDWARE:-AX650}"
NPU_MODE="${NPU_MODE:-NPU3}"

mkdir -p "${AXMODEL_DIR}"

CONFIG="config_encoder_nolog_check0${BUILD_SUFFIX}.json"
BUILD_DIR="build-encoder_nolog-check0${BUILD_SUFFIX}"

cat > "${CONFIG}" << EOF
{
    "model_type": "ONNX",
    "npu_mode": "${NPU_MODE}",
    "input": "${MODEL_DIR}/encoder_core.onnx",
    "output_name": "${OUTPUT_NAME}",
    "output_dir": "./${BUILD_DIR}",
    "target_hardware": "${TARGET_HARDWARE}",
    "onnx_opt": {"disable_onnx_optimization": false, "enable_onnxsim": true},
    "quant": {
        "input_configs": [
            {"tensor_name": "cat_tokens", "calibration_dataset": "${CALIB_DIR}/encoder_core/cat_tokens.tar.gz", "calibration_format": "Numpy", "calibration_size": ${CALIBRATION_SIZE}}
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

pulsar2 build --config "${CONFIG}" --input_shapes "cat_tokens:1x${MAX_TOKENS}" 2>&1 | tee "${BUILD_DIR}.log"
cp "${BUILD_DIR}/${OUTPUT_NAME}" "${AXMODEL_DIR}/"
ls -lh "${AXMODEL_DIR}/${OUTPUT_NAME}"
