#!/usr/bin/env python3
"""ZipVoice static ONNX inference using encoder_core + four decoder parts."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import infer_onnx_static as infer_base  # noqa: E402
from onnx_zipvoice_decoder4_runtime import Decoder4StaticOnnxZipVoiceRuntime  # noqa: E402


infer_base.StaticOnnxZipVoiceRuntime = Decoder4StaticOnnxZipVoiceRuntime


if __name__ == "__main__":
    infer_base.main()
