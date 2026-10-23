#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ZIPVOICE_AXMODEL_DIR="${ROOT_DIR}/model_convert/zipvoice_ax650"
ZIPVOICE_DISTILL_AXMODEL_DIR="${ROOT_DIR}/model_convert/zipvoice_distill_ax650"
mkdir -p "${ZIPVOICE_AXMODEL_DIR}" "${ZIPVOICE_DISTILL_AXMODEL_DIR}"

MODEL_DIR="${ROOT_DIR}/models/decoder4_static" \
CALIB_DIR="${ROOT_DIR}/model_convert/calib_data_decoder4_rich" \
AXMODEL_DIR="${ZIPVOICE_AXMODEL_DIR}" \
TARGET_HARDWARE="${TARGET_HARDWARE:-AX650}" \
NPU_MODE="${NPU_MODE:-NPU3}" \
CHECK_LEVEL="${CHECK_LEVEL:-0}" \
BUILD_SUFFIX="-1024-ax650-npu3" \
OUTPUT_NAME="encoder.axmodel" \
"${ROOT_DIR}/utils/quant_encoder_nolog_check0.sh"

MODEL_DIR="${ROOT_DIR}/models/decoder4_static" \
CALIB_DIR="${ROOT_DIR}/model_convert/calib_data_decoder4_rich" \
AXMODEL_DIR="${ZIPVOICE_AXMODEL_DIR}" \
MAX_FEAT_LEN="${MAX_FEAT_LEN:-1024}" \
DECODER_BATCH="${DECODER_BATCH:-2}" \
MASK_INPUT_NAME="${MASK_INPUT_NAME:-padding_mask2}" \
MASK_SHAPE="${MASK_SHAPE:-2x1024}" \
TARGET_HARDWARE="${TARGET_HARDWARE:-AX650}" \
NPU_MODE="${NPU_MODE:-NPU3}" \
CHECK_LEVEL="${CHECK_LEVEL:-0}" \
BUILD_SUFFIX="-1024-ax650-npu3" \
OUTPUT_NAME_PART0="decoder_part0.axmodel" \
OUTPUT_NAME_PART1="decoder_part1.axmodel" \
OUTPUT_NAME_PART2="decoder_part2.axmodel" \
OUTPUT_NAME_PART3="decoder_part3.axmodel" \
"${ROOT_DIR}/utils/quant_decoder4_check0.sh"

MODEL_DIR="${ROOT_DIR}/models/decoder4_static_distill_1024" \
CALIB_DIR="${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024" \
AXMODEL_DIR="${ZIPVOICE_DISTILL_AXMODEL_DIR}" \
TARGET_HARDWARE="${TARGET_HARDWARE:-AX650}" \
NPU_MODE="${NPU_MODE:-NPU3}" \
CHECK_LEVEL="${CHECK_LEVEL:-0}" \
BUILD_SUFFIX="-distill-1024-ax650-npu3" \
OUTPUT_NAME="encoder.axmodel" \
"${ROOT_DIR}/utils/quant_encoder_nolog_check0.sh"

MODEL_DIR="${ROOT_DIR}/models/decoder4_static_distill_1024" \
CALIB_DIR="${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024" \
AXMODEL_DIR="${ZIPVOICE_DISTILL_AXMODEL_DIR}" \
MAX_FEAT_LEN="${MAX_FEAT_LEN:-1024}" \
DECODER_BATCH="${DECODER_BATCH:-1}" \
MASK_INPUT_NAME="${MASK_INPUT_NAME:-padding_mask}" \
MASK_SHAPE="${MASK_SHAPE:-1x1024}" \
TARGET_HARDWARE="${TARGET_HARDWARE:-AX650}" \
NPU_MODE="${NPU_MODE:-NPU3}" \
CHECK_LEVEL="${CHECK_LEVEL:-0}" \
BUILD_SUFFIX="-distill-1024-ax650-npu3" \
OUTPUT_NAME_PART0="decoder_part0.axmodel" \
OUTPUT_NAME_PART1="decoder_part1.axmodel" \
OUTPUT_NAME_PART2="decoder_part2.axmodel" \
OUTPUT_NAME_PART3="decoder_part3.axmodel" \
"${ROOT_DIR}/utils/quant_decoder4_check0.sh"

find "${ZIPVOICE_AXMODEL_DIR}" "${ZIPVOICE_DISTILL_AXMODEL_DIR}" -maxdepth 1 -type f -name '*.axmodel' | sort
