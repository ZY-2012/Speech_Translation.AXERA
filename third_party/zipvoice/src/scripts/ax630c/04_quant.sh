#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export WORK_TMP="${WORK_TMP:-${ROOT_DIR}/.work_tmp}"
export TMPDIR="${TMPDIR:-${WORK_TMP}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_TMP}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_TMP}/xdg_cache}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

AXMODEL_DIR="${ROOT_DIR}/model_convert/zipvoice_distill_ax630c"
mkdir -p "${AXMODEL_DIR}"

echo "=== AX630C: quantize encoder ==="
MODEL_DIR="${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c" \
CALIB_DIR="${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024" \
AXMODEL_DIR="${AXMODEL_DIR}" \
TARGET_HARDWARE="AX620E" \
NPU_MODE="NPU2" \
CHECK_LEVEL="${CHECK_LEVEL:-0}" \
BUILD_SUFFIX="-distill-1024-ax630c-npu2" \
OUTPUT_NAME="encoder.axmodel" \
"${ROOT_DIR}/utils/quant_encoder_nolog_check0.sh"

echo "=== AX630C: quantize decoder part0/part1/part2 single-chain base ==="
MODEL_DIR="${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c" \
CALIB_DIR="${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024" \
AXMODEL_DIR="${AXMODEL_DIR}" \
MAX_FEAT_LEN="${MAX_FEAT_LEN:-1024}" \
DECODER_BATCH="1" \
MASK_INPUT_NAME="padding_mask" \
MASK_SHAPE="1x1024" \
TARGET_HARDWARE="AX620E" \
NPU_MODE="NPU2" \
CHECK_LEVEL="${CHECK_LEVEL:-0}" \
BUILD_SUFFIX="-distill-1024-ax630c-npu2" \
PIECES="fm_decoder_part0,fm_decoder_part1,fm_decoder_part2" \
OUTPUT_NAME_PART0="decoder_part0.axmodel" \
OUTPUT_NAME_PART1="decoder_part1.axmodel" \
OUTPUT_NAME_PART2="decoder_part2.axmodel" \
"${ROOT_DIR}/utils/quant_decoder4_check0.sh"

echo "=== AX630C: generate part3-four Pulsar2 configs ==="
python3 "${ROOT_DIR}/utils/model_convert/make_part3four_quant_configs.py"

cd "${ROOT_DIR}/model_convert"

echo "=== AX630C: quantize fm_decoder_part3_0 ==="
pulsar2 build \
  --config config_fm_decoder_part3_0_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p2:1024x1x512;time_hidden:1x192;padding_mask:1x1024" \
  2>&1 | tee build-fm_decoder_part3_0-check0-distill-rich-four-u16-powu8-ax630c.log

echo "=== AX630C: quantize fm_decoder_part3_1 ==="
pulsar2 build \
  --config config_fm_decoder_part3_1_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p30:1024x1x512;time_hidden:1x192;padding_mask:1x1024" \
  2>&1 | tee build-fm_decoder_part3_1-check0-distill-rich-four-u16-powu8-ax630c.log

echo "=== AX630C: quantize fm_decoder_part3_2 ==="
pulsar2 build \
  --config config_fm_decoder_part3_2_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p31:1024x1x512;time_hidden:1x192;padding_mask:1x1024" \
  2>&1 | tee build-fm_decoder_part3_2-check0-distill-rich-four-u16-powu8-ax630c.log

echo "=== AX630C: quantize fm_decoder_part3_3 ==="
pulsar2 build \
  --config config_fm_decoder_part3_3_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p32:1024x1x512;time_hidden:1x192;padding_mask:1x1024" \
  2>&1 | tee build-fm_decoder_part3_3-check0-distill-rich-four-u16-powu8-ax630c.log

echo "=== AX630C: update final distill package ==="
cp \
  "${ROOT_DIR}/model_convert/build-fm_decoder_part3_0-check0-distill-rich-four-u16-powu8-ax630c/decoder_part3_0.axmodel" \
  "${ROOT_DIR}/model_convert/build-fm_decoder_part3_1-check0-distill-rich-four-u16-powu8-ax630c/decoder_part3_1.axmodel" \
  "${ROOT_DIR}/model_convert/build-fm_decoder_part3_2-check0-distill-rich-four-u16-powu8-ax630c/decoder_part3_2.axmodel" \
  "${ROOT_DIR}/model_convert/build-fm_decoder_part3_3-check0-distill-rich-four-u16-powu8-ax630c/decoder_part3_3.axmodel" \
  "${AXMODEL_DIR}/"

cp "${ROOT_DIR}/model_convert/decoder4_split_manifest_part3four.json" \
  "${AXMODEL_DIR}/decoder4_split_manifest.json"

echo ""
echo "=== AX630C final package ==="
find "${AXMODEL_DIR}" -maxdepth 1 -type f -name '*.axmodel' -o -name 'decoder4_split_manifest.json' | sort
