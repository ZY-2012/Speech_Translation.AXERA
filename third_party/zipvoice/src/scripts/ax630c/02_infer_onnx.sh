#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=== AX630C: verify ZipVoice distill static ONNX ==="

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c}" \
OUTPUT_WAV="${OUTPUT_WAV:-${ROOT_DIR}/outputs/decoder4_static_distill_1024_ax630c_onnx.wav}" \
"${ROOT_DIR}/utils/infer_decoder4_static_onnx_ax630c.sh"
