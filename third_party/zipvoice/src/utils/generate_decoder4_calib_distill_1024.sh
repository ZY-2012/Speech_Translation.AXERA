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

python3 utils/model_convert/calibration_attn_loop_decoder4_rich.py \
  --model-dir "${MODEL_DIR:-${ROOT_DIR}/models/decoder4_static_distill_1024}" \
  --output-dir "${OUTPUT_DIR:-${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024}" \
  --prompt-wav "${ROOT_DIR}/assets/calib_audio/zipvoice_prompt.wav" \
  --prompt-text "I am a prompt for voice cloning." \
  --prompt-wav "${ROOT_DIR}/assets/calib_audio/moss_zh_3.wav" \
  --prompt-text "这是一段用于语音克隆的提示音频，我们希望模型学习说话人的声音和节奏。" \
  --num-calib-texts "${NUM_CALIB_TEXTS:-4}" \
  --num-step "${NUM_STEP:-4}" \
  --guidance-scale "${GUIDANCE_SCALE:-3.0}" \
  --seed "${SEED:-1234}" \
  --speed-list "${SPEED_LIST:-1.0,0.85,1.15,1.0}"
