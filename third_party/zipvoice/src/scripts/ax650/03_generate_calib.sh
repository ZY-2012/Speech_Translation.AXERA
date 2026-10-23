#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

"${ROOT_DIR}/utils/generate_decoder4_calib.sh"
"${ROOT_DIR}/utils/generate_decoder4_calib_distill_1024.sh"
