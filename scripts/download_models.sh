#!/usr/bin/env bash
# Download models used by the default SenseVoice + TEN-VAD + Opus-MT + ZipVoice path.
# Qwen3 / Zipformer / FireRedPunc are optional and are not downloaded here.
# Usage: bash scripts/download_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf not found. Install it with: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p models/asr_chain models/asr_official/sensevoice_ax650 \
  models/opus_mt models/zipvoice models/.download

echo "[1/3] SenseVoice + TEN-VAD"
hf download AXERA-TECH/models-for-asr-chain \
  --include "asr/*" "vad/*" --local-dir models/asr_chain

echo "[2/3] streaming SenseVoice"
hf download HY-2012/Tmp_Models \
  --include "*streaming_sensevoice.axmodel" \
  --local-dir models/.download/stream
STREAM_MODEL="$(find models/.download/stream -name streaming_sensevoice.axmodel -print -quit)"
if [[ -z "$STREAM_MODEL" ]]; then
  echo "streaming_sensevoice.axmodel not found in HF download" >&2
  exit 1
fi
cp "$STREAM_MODEL" models/asr_chain/asr/sensevoice/streaming_sensevoice.axmodel
mkdir -p models/asr_official/sensevoice_ax650/sensevoice
cp -r models/asr_chain/asr/sensevoice/. models/asr_official/sensevoice_ax650/sensevoice/
cp -r models/asr_chain/asr/sensevoice/. models/asr_official/sensevoice_ax650/
rm -rf models/.download/stream

echo "[3/3] Opus-MT + ZipVoice"
hf download HY-2012/Opus-MT.AXERA --local-dir models/opus_mt
hf download AXERA-TECH/ZipVoice.AXERA --local-dir models/zipvoice

cat <<'EOF'
Done.

Optional (not downloaded by this script):
  Qwen3-0.6B:  hf download AXERA-TECH/Qwen3-0.6B-GPTQ-Int4 --local-dir models/qwen3-0.6b
  FireRedPunc: put ../FireRedPunc.AXERA/models/punc.axmodel in place
  Zipformer:   put ../ZF/streaming/release/models in place
EOF
