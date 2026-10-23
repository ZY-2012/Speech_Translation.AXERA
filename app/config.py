"""Configuration loading and path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

_DEFAULTS = {
    "server": {"host": "0.0.0.0", "port": 8090, "max_sessions": 1},
    "audio": {
        "sample_rate": 16000,
        "vad_hop_ms": 16,
        "pre_roll_ms": 240,
        "post_roll_ms": 160,
        "min_speech_ms": 400,
        "trailing_silence_ms": 500,
        "max_segment_ms": 6000,
    },
    "vad": {
        "backend": "ten",
        "energy_threshold": 0.012,
        "library": "third_party/ten_vad/libten_vad.so",
        "model_dir": "models/asr_chain/vad/ten-vad",
        "threshold": 0.3,
    },
    "asr": {
        "stream_enabled": True,
        "stream_worker": "third_party/stream_asr/bin/stream-asr-worker",
        "stream_model_dir": "models/asr_chain/asr",
        "language": "auto",
        "offline_enabled": False,
        "offline_model_dir": "models/asr_offline",
        "offline_language": "auto",
        "withitn": True,
        "max_chars": 220,
    },
    "llm": {
        "enabled": True,
        "worker": "third_party/llm/qwen3-worker",
        "model_dir": "models/llm/qwen3-1.7b",
        "system_prompt": "You are a professional interpreter. Output only the translation.",
        "max_tokens": 160,
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout_s": 30,
        "direction": "auto",
    },
    "tts": {
        "enabled": True,
        "runtime": "third_party/pocket_tts/board_pocket_tts_axera.py",
        "model_dir": "models/pocket-tts/models",
        "reference": "models/pocket-tts/models/Vivian.wav",
        "threads": 4,
        "prefill_threads": 8,
        "max_tokens_per_chunk": 48,
        "gap_ms": 120,
        "stream": True,
        "npu": True,
        "clause_min_chars": 6,
        "clause_max_chars": 26,
        "commit_min_chars": 6,
        "commit_max_chars": 26,
        "max_pending_mt": 8,
        "min_final_chars": 8,
        "pause_stream_during_tts": False,
    },
    "runtime": {
        "work_dir": ".work",
        "log_dir": ".work/logs",
        "npu_lock_file": ".work/npu2.lock",
        "barge_in": False,
        "preload": True,
    },
}


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve(value: str) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def load(path: str | os.PathLike | None = None) -> dict:
    data = copy.deepcopy(_DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = _merge(data, yaml.safe_load(f) or {})

    # Resolve every filesystem-ish value explicitly.
    data["runtime"]["work_dir"] = _resolve(data["runtime"]["work_dir"])
    data["runtime"]["log_dir"] = _resolve(data["runtime"]["log_dir"])
    data["runtime"]["npu_lock_file"] = _resolve(data["runtime"]["npu_lock_file"])
    for section, key in (
        ("vad", "library"), ("vad", "model_dir"),
        ("asr", "stream_worker"), ("asr", "stream_model_dir"),
        ("asr", "offline_model_dir"),
        ("llm", "worker"), ("llm", "model_dir"),
        ("tts", "runtime"), ("tts", "model_dir"), ("tts", "reference"),
    ):
        data[section][key] = _resolve(data[section][key])

    os.makedirs(data["runtime"]["work_dir"], exist_ok=True)
    os.makedirs(data["runtime"]["log_dir"], exist_ok=True)
    return data
