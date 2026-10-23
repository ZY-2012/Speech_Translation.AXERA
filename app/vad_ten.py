"""TEN-VAD (ctypes) and the utterance segmenter used by the web session."""
from __future__ import annotations

import ctypes
import math
import os
from collections import deque
from pathlib import Path

import numpy as np


class TenVad:
    """Thin ctypes wrapper.  The .so loads ten-vad.axmodel from its cwd."""

    def __init__(self, lib_path: str, model_dir: str, hop: int = 256,
                 threshold: float = 0.3):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.ten_vad_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_float]
        self.lib.ten_vad_create.restype = ctypes.c_int
        self.lib.ten_vad_process.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
        self.lib.ten_vad_process.restype = ctypes.c_int
        self.lib.ten_vad_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.handle = ctypes.c_void_p()
        old = os.getcwd()
        os.chdir(str(model_dir))
        try:
            if self.lib.ten_vad_create(ctypes.byref(self.handle), hop, threshold) != 0:
                raise RuntimeError("ten_vad_create failed")
        finally:
            os.chdir(old)

    def process(self, pcm: np.ndarray):
        prob = ctypes.c_float(0.0)
        flag = ctypes.c_int(0)
        arr = np.ascontiguousarray(pcm, dtype=np.int16)
        status = self.lib.ten_vad_process(
            self.handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            arr.size, ctypes.byref(prob), ctypes.byref(flag))
        if status != 0:
            raise RuntimeError(f"ten_vad_process failed: {status}")
        return float(prob.value), bool(flag.value)

    def close(self):
        if self.handle:
            self.lib.ten_vad_destroy(ctypes.byref(self.handle))
            self.handle = ctypes.c_void_p()


class EnergyVad:
    """CPU VAD for the web path (keeps the audio thread off the NPU)."""

    def __init__(self, threshold: float = 0.012, hangover_hops: int = 10):
        self.threshold = threshold
        self.noise = 0.003
        self.hangover = max(0, hangover_hops)
        self._hang = 0

    def process(self, pcm: np.ndarray):
        x = np.asarray(pcm, dtype=np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        self.noise = 0.95 * self.noise + 0.05 * rms
        voiced = rms > max(self.threshold, self.noise * 1.8)
        if voiced:
            self._hang = self.hangover
        elif self._hang > 0:
            self._hang -= 1
            voiced = True
        prob = min(1.0, rms / max(1e-4, self.threshold * 2))
        return prob, voiced


def make_vad(cfg: dict, mock: bool = False):
    hop = int(cfg["audio"]["sample_rate"] * cfg["audio"]["vad_hop_ms"] / 1000)
    backend = cfg["vad"].get("backend", "ten")
    if mock or backend == "energy":
        # CPU VAD: keeps the audio thread responsive while Qwen/TTS use the
        # NPU.  The TEN-VAD ctypes call is synchronous and competes with them.
        return EnergyVad(threshold=float(cfg["vad"].get("energy_threshold", 0.012)))
    return TenVad(cfg["vad"]["library"], cfg["vad"]["model_dir"], hop,
                  float(cfg["vad"]["threshold"]))


class Segmenter:
    """Hop-level VAD state machine with pre/post roll and max-length splitting."""

    def __init__(self, cfg: dict, vad, on_start=None, on_hop=None, on_end=None):
        audio = cfg["audio"]
        self.rate = int(audio["sample_rate"])
        self.hop = int(self.rate * audio["vad_hop_ms"] / 1000)
        ms = audio["vad_hop_ms"]
        self.pre_hops = max(1, int(round(audio["pre_roll_ms"] / ms)))
        self.post_hops = max(1, int(round(audio["post_roll_ms"] / ms)))
        self.min_hops = max(1, int(round(audio["min_speech_ms"] / ms)))
        self.trail_hops = max(1, int(round(audio["trailing_silence_ms"] / ms)))
        self.max_hops = max(1, int(round(audio["max_segment_ms"] / ms)))
        self.vad = vad
        self.on_start = on_start
        self.on_hop = on_hop
        self.on_end = on_end

        self._pending = np.zeros(0, dtype=np.int16)
        self._pre = deque(maxlen=self.pre_hops)
        self._buf: list[np.ndarray] = []
        self._in_speech = False
        self._voiced_run = 0
        self._silence_run = 0
        self._voiced_hops = 0
        self._last_voiced = 0
        self._start_sample = 0
        self._samples_seen = 0
        self._seg_index = 0

    def push(self, pcm: np.ndarray):
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if pcm.size == 0:
            return
        if self._pending.size:
            pcm = np.concatenate([self._pending, pcm])
        n = pcm.size // self.hop
        for i in range(n):
            hop = pcm[i * self.hop:(i + 1) * self.hop]
            self._process_hop(hop)
        self._pending = pcm[n * self.hop:].copy()

    def _process_hop(self, hop: np.ndarray):
        index = self._samples_seen
        self._samples_seen += hop.size
        _, voiced = self.vad.process(hop)

        if not self._in_speech:
            self._pre.append(hop)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= 2:
                self._in_speech = True
                self._seg_index += 1
                self._buf = list(self._pre)
                self._start_sample = index - (len(self._buf) - 1) * self.hop
                self._voiced_hops = self._voiced_run
                self._silence_run = 0
                self._last_voiced = len(self._buf)
                if self.on_start:
                    self.on_start(self._seg_index, self._start_sample,
                                  np.concatenate(self._buf) if self._buf else hop)
            return

        self._buf.append(hop)
        if voiced:
            self._silence_run = 0
            self._voiced_hops += 1
            self._last_voiced = len(self._buf)
        else:
            self._silence_run += 1

        if self.on_hop:
            self.on_hop(hop)

        force = len(self._buf) >= self.max_hops
        if self._silence_run >= self.trail_hops or force:
            self._close(force=force)

    def _close(self, force: bool = False):
        if not self._in_speech:
            return
        end_idx = len(self._buf)
        if not force:
            end_idx = min(len(self._buf), self._last_voiced + self.post_hops)
        segment = np.concatenate(self._buf[:end_idx]) if end_idx else np.zeros(0, np.int16)
        end_sample = self._start_sample + segment.size
        speech_ms = self._voiced_hops * (self.hop * 1000 / self.rate)
        self._in_speech = False
        self._buf = []
        self._silence_run = 0
        self._voiced_run = 0
        self._voiced_hops = 0
        self._last_voiced = 0
        # Always notify the end, even for a short burst: the session must
        # flush the streaming ASR and close the TTS uid in lockstep with VAD.
        if segment.size and self.on_end:
            self.on_end(segment, self._start_sample, end_sample, speech_ms / 1000)
        # keep the tail as pre-roll for the next utterance
        self._pre.clear()

    def flush(self):
        if self._in_speech:
            self._close(force=False)

    @property
    def in_speech(self) -> bool:
        return self._in_speech
