#!/usr/bin/env bash
# Export Vocos ONNX model
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Exporting Vocos ONNX ==="
python3 python/export_vocos_onnx.py

echo "=== ONNX files ==="
ls -lh model_convert/vocoder_onnx/*.onnx 2>/dev/null || echo "No ONNX files found"
