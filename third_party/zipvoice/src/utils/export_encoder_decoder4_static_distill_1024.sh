#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export WORK_TMP="${WORK_TMP:-${ROOT_DIR}/.work_tmp}"
export TMPDIR="${TMPDIR:-${WORK_TMP}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_TMP}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_TMP}/xdg_cache}"
export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/zipvoice_distill}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-model.pt}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MAX_FEAT_LEN="${MAX_FEAT_LEN:-1024}"
OPSET="${OPSET:-13}"
REEXPORT_RAW="${REEXPORT_RAW:-0}"

RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-${WORK_TMP}/attn_loop_raw_distill_1024}"
FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-${WORK_TMP}/full_static_distill_1024}"
DECODER4_OUTPUT_DIR="${DECODER4_OUTPUT_DIR:-${ROOT_DIR}/models/decoder4_static_distill_1024}"
SPLIT_WORK_DIR="${SPLIT_WORK_DIR:-${WORK_TMP}/split_decoder4_distill_1024}"

mkdir -p "${RAW_OUTPUT_DIR}" "${FULL_OUTPUT_DIR}" "${DECODER4_OUTPUT_DIR}" "${SPLIT_WORK_DIR}"

echo "Export distill nolog encoder + attn-loop sigmoid-bump full decoder + decoder4 static ONNX"
echo "Model dir:           ${MODEL_DIR}"
echo "Work static dir:     ${FULL_OUTPUT_DIR}"
echo "Decoder4 ONNX dir:   ${DECODER4_OUTPUT_DIR}"

if [ "${REEXPORT_RAW}" = "1" ] || \
   [ ! -f "${RAW_OUTPUT_DIR}/encoder_core.onnx" ] || \
   [ ! -f "${RAW_OUTPUT_DIR}/fm_decoder.onnx" ]; then
  python3 utils/model_convert/export_static_attention_loop_onnx.py \
    --model-dir "${MODEL_DIR}" \
    --checkpoint-name "${CHECKPOINT_NAME}" \
    --output-dir "${RAW_OUTPUT_DIR}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-feat-len "${MAX_FEAT_LEN}" \
    --opset "${OPSET}" \
    --distill
else
  echo "Reusing raw distill attention-loop ONNX in ${RAW_OUTPUT_DIR}"
fi

cp "${RAW_OUTPUT_DIR}/runtime_config.json" "${FULL_OUTPUT_DIR}/runtime_config.json"

python3 utils/model_convert/rewrite_swoosh_nolog_onnx.py \
  --input "${RAW_OUTPUT_DIR}/encoder_core.onnx" \
  --output "${FULL_OUTPUT_DIR}/encoder_core.onnx.nolog_raw.onnx"

python3 utils/model_convert/simplify_onnx_file.py \
  --input "${FULL_OUTPUT_DIR}/encoder_core.onnx.nolog_raw.onnx" \
  --output "${FULL_OUTPUT_DIR}/encoder_core.onnx"

python3 utils/model_convert/rewrite_swoosh_nolog_onnx.py \
  --input "${RAW_OUTPUT_DIR}/fm_decoder.onnx" \
  --output "${FULL_OUTPUT_DIR}/fm_decoder.onnx.sigmoid_bump_raw.onnx" \
  --approx sigmoid-bump

python3 utils/model_convert/simplify_onnx_file.py \
  --input "${FULL_OUTPUT_DIR}/fm_decoder.onnx.sigmoid_bump_raw.onnx" \
  --output "${FULL_OUTPUT_DIR}/fm_decoder.onnx"

python3 utils/model_convert/split_static_decoder4.py \
  --source-model-dir "${FULL_OUTPUT_DIR}" \
  --output-dir "${DECODER4_OUTPUT_DIR}" \
  --work-dir "${SPLIT_WORK_DIR}"

python3 utils/model_convert/fix_decoder2_cfg_scale_shape.py --model "${DECODER4_OUTPUT_DIR}/fm_decoder_part0.onnx"
python3 utils/model_convert/fix_decoder2_cfg_scale_shape.py --model "${DECODER4_OUTPUT_DIR}/fm_decoder_part3.onnx"

cp "${FULL_OUTPUT_DIR}/encoder_core.onnx" "${DECODER4_OUTPUT_DIR}/encoder_core.onnx"
cp "${FULL_OUTPUT_DIR}/runtime_config.json" "${DECODER4_OUTPUT_DIR}/runtime_config.json"

ls -lh \
  "${DECODER4_OUTPUT_DIR}/encoder_core.onnx" \
  "${DECODER4_OUTPUT_DIR}/fm_decoder_part0.onnx" \
  "${DECODER4_OUTPUT_DIR}/fm_decoder_part1.onnx" \
  "${DECODER4_OUTPUT_DIR}/fm_decoder_part2.onnx" \
  "${DECODER4_OUTPUT_DIR}/fm_decoder_part3.onnx"
