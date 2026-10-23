#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Export both supported 1024-length model variants:
# 1. standard ZipVoice
# 2. ZipVoice distill
MODEL_DIR="${ZIPVOICE_MODEL_DIR:-${ROOT_DIR}/models/zipvoice}" \
"${ROOT_DIR}/utils/export_encoder_decoder4_static.sh"

MODEL_DIR="${ZIPVOICE_DISTILL_MODEL_DIR:-${ROOT_DIR}/models/zipvoice_distill}" \
"${ROOT_DIR}/utils/export_encoder_decoder4_static_distill_1024.sh"
