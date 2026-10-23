#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=== AX630C: export ZipVoice distill static ONNX ==="
echo "Root: ${ROOT_DIR}"

"${ROOT_DIR}/utils/export_encoder_decoder4_static_distill_1024_ax630c.sh"

echo ""
echo "=== AX630C ONNX files ==="
ls -lh \
  "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c/encoder_core.onnx" \
  "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c/fm_decoder_part0.onnx" \
  "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c/fm_decoder_part1.onnx" \
  "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c/fm_decoder_part2.onnx" \
  "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c/fm_decoder_part3.onnx"
