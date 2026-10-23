#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Infer only the ONNX models that will be quantized: encoder + 4 decoder.
"${ROOT_DIR}/utils/infer_decoder4_static_onnx.sh"

MODEL_DIR="${ROOT_DIR}/models/decoder4_static_distill_1024" \
OUTPUT_WAV="${ROOT_DIR}/outputs/decoder4_static_distill_1024_onnx.wav" \
NUM_STEP="${NUM_STEP_DISTILL:-4}" \
GUIDANCE_SCALE="${GUIDANCE_SCALE_DISTILL:-3.0}" \
"${ROOT_DIR}/utils/infer_decoder4_static_onnx.sh"
