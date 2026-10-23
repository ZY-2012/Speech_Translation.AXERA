#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=== AX630C: generate rich distill calibration data ==="
"${ROOT_DIR}/utils/generate_decoder4_calib_distill_1024_ax630c.sh"

echo ""
echo "=== AX630C: split part3 into four ONNX models ==="
python3 "${ROOT_DIR}/utils/model_convert/split_fm_decoder_part3_four.py" \
  --input "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c/fm_decoder_part3.onnx" \
  --output-dir "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c_part3four" \
  --work-dir "${ROOT_DIR}/.work_tmp/split_part3four_ax630c"

echo ""
echo "=== AX630C: generate part3-four calibration data ==="
python3 "${ROOT_DIR}/utils/model_convert/generate_part3four_calib.py" \
  --model-dir "${ROOT_DIR}/models/decoder4_static_distill_1024_ax630c_part3four" \
  --source-calib-dir "${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024/fm_decoder_part3" \
  --output-dir "${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024_part3four"

echo ""
echo "=== AX630C calibration data ==="
du -sh \
  "${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024" \
  "${ROOT_DIR}/model_convert/calib_data_decoder4_rich_distill_1024_part3four"
