#!/usr/bin/env bash
# Generate vocoder calibration data
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Generating vocoder calibration data ==="
python3 python/generate_vocoder_calib.py

echo ""
echo "=== Calibration data ==="
du -sh model_convert/vocoder_onnx/calib_data/ 2>/dev/null || echo "No calib data found"
find model_convert/vocoder_onnx/calib_data -name "*.tar.gz" -exec ls -lh {} \;
