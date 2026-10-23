#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export WORK_TMP="${WORK_TMP:-${ROOT_DIR}/.work_tmp}"
export TMPDIR="${TMPDIR:-${WORK_TMP}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_TMP}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_TMP}/xdg_cache}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" outputs

export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}:${PYTHONPATH:-}"

python3 python/infer_onnx_decoder4_static.py \
  --model-dir "${MODEL_DIR:-${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c}" \
  --prompt-wav "${PROMPT_WAV:-${ROOT_DIR}/assets/moss_prompts/zh_1_4p5s.wav}" \
  --prompt-text "${PROMPT_TEXT:-不管怎么样我和汤姆还是要感谢贝尔卡金的援手}" \
  --text "${TEXT:-今天午后天气很好，我打开窗户，听见远处有人聊天，水杯也轻轻晃了一下。}" \
  --output-wav "${OUTPUT_WAV:-${ROOT_DIR}/outputs/decoder4_static_distill_1024_ax630c_onnx.wav}" \
  --num-step "${NUM_STEP:-16}" \
  --guidance-scale "${GUIDANCE_SCALE:-1.0}" \
  --seed "${SEED:-42}" \
  --num-threads "${NUM_THREADS:-4}" \
  --vocoder-path "${VOCODER_PATH:-${ROOT_DIR}/models/vocos-mel-24khz}"
