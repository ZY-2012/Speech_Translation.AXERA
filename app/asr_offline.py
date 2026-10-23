"""Resident non-streaming SenseVoice ASR via the official libax_asr_api.so.

The board is fed one VAD clause at a time.  The offline model returns
punctuated text, which is then split into source clauses before MT.
"""
from __future__ import annotations

import ctypes
import os
import traceback

import numpy as np

AX_SENSEVOICE = 4


class OfflineAsr:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conf = cfg["asr"]
        self.lib = None
        self.handle = None
        self._ready = False

    def load(self):
        if self._ready:
            return
        self.lib = ctypes.CDLL(self.conf["library"])
        self.lib.AX_ASR_Init.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.AX_ASR_Init.restype = ctypes.c_void_p
        self.lib.AX_ASR_Uninit.argtypes = [ctypes.c_void_p]
        self.lib.AX_ASR_Uninit.restype = None
        self.lib.AX_ASR_RunPCM.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p)]
        self.lib.AX_ASR_RunPCM.restype = ctypes.c_int
        self.lib.AX_ASR_Free.argtypes = [ctypes.c_char_p]
        self.lib.AX_ASR_Free.restype = None
        model_dir = self.conf["model_dir"]
        os.environ.setdefault("AX_SENSEVOICE_SHARE_DIR", model_dir)
        self.handle = self.lib.AX_ASR_Init(AX_SENSEVOICE, model_dir.encode())
        if not self.handle:
            raise RuntimeError("AX_ASR_Init failed")
        self._ready = True
        print("[asr] offline SenseVoice ready", flush=True)

    def ready(self) -> bool:
        return self._ready and self.handle is not None

    def run(self, pcm_bytes: bytes) -> str:
        self.load()
        pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        if pcm.size == 0:
            return ""
        arr = np.ascontiguousarray(pcm, dtype=np.float32)
        out = ctypes.c_char_p()
        language = (self.conf.get("language") or "auto").encode()
        try:
            rc = self.lib.AX_ASR_RunPCM(
                self.handle,
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                arr.size, int(self.cfg["audio"]["sample_rate"]),
                language, ctypes.byref(out))
            if rc != 0:
                print(f"[asr] RunPCM failed rc={rc}", flush=True)
                return ""
            text = out.value.decode("utf-8", "replace") if out.value else ""
            return text.strip()
        finally:
            if out.value:
                self.lib.AX_ASR_Free(out)

    def close(self):
        if self.handle is not None and self.lib is not None:
            try:
                self.lib.AX_ASR_Uninit(self.handle)
            except Exception:
                traceback.print_exc()
            self.handle = None
        self._ready = False
