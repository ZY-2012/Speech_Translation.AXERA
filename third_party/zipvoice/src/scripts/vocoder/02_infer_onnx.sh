#!/usr/bin/env bash
# ONNX inference verification for vocoder
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_DIR="${ROOT_DIR}/model_convert/vocoder_onnx"
FULL_MODEL="${MODEL_DIR}/vocos_full_B1_T620.onnx"

if [ ! -f "${FULL_MODEL}" ]; then
    echo "ERROR: ${FULL_MODEL} not found. Run scripts/vocoder/01_export_onnx.sh first."
    exit 1
fi

echo "=== Verifying vocoder ONNX inference ==="
python3 << PYEOF
import sys, numpy as np
import onnxruntime as ort

T, N_MELS = 620, 100
mel = np.random.randn(1, N_MELS, T).astype(np.float32) * 0.1
model_path = '${FULL_MODEL}'

print(f'Loading {model_path}...')
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print('Running inference...')
outputs = session.run(None, {'mel': mel})

for i, name in enumerate([o.name for o in session.get_outputs()]):
    print(f'  output[{i}] {name}: shape={outputs[i].shape}, dtype={outputs[i].dtype}')

print(f'  real range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]')
print(f'  imag range: [{outputs[1].min():.4f}, {outputs[1].max():.4f}]')
print('ONNX inference OK')
PYEOF

echo ""
echo "=== Verification complete ==="
ls -lh "${FULL_MODEL}"
