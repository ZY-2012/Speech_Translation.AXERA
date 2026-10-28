"""Resident streaming SenseVoice worker client (continuous audio + epochs).

Uses the AX650 `stream-asr-worker` protocol:
  stdin:  uint32 n + n bytes S16LE PCM;  n == 0 flushes the epoch
  fd:     'R' ready
          'T' uint32 flags + uint32 coverage_samples + UTF-8 text
              flags bit0 = partial, bit1 = final
          'E' error

The ASR model is continuous; VAD only decides when to flush an epoch, so
sentence boundaries do not require re-running an offline model on each VAD
fragment.
"""
from __future__ import annotations

import os
import struct
import subprocess
import threading
import time
from pathlib import Path


class StreamAsr:
    def __init__(self, cfg: dict, on_partial=None):
        self.cfg = cfg
        self.conf = cfg["asr"]
        self.on_partial = on_partial
        self._proc = None
        self._rfd = -1
        self._log = None
        self._reader = None
        self._ready_event = threading.Event()
        self._final_event = threading.Event()
        self._final_text = ""
        self._partial = ""
        self._ready = False
        self._lock = threading.Lock()

    def start(self, wait: bool = True) -> bool:
        if self._ready:
            return True
        root = Path(self.cfg["runtime"]["work_dir"]).parent
        log_dir = Path(self.cfg["runtime"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(log_dir / "stream_asr.log", "ab", buffering=0)
        r, w = os.pipe()
        env = dict(os.environ)
        # Python workers (the optional Zipformer backend) run under the
        # system Python 3.10; do not leak the main 3.11 PYTHONPATH/pylibs.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["LD_LIBRARY_PATH"] = (str(root / "third_party/asr/lib") +
                                  ":/soc/lib:/opt/lib")
        env.setdefault("AX_SENSEVOICE_SHARE_DIR", str(root / self.conf["stream_model_dir"]))
        args = [
            str(root / self.conf["stream_worker"]),
            "--model-dir", str(root / self.conf["stream_model_dir"]),
            "--language", self.conf.get("language", "zh"),
            "--response-fd", str(w),
        ]
        self._log.write(("\n=== start " + " ".join(args) + " ===\n").encode())
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                      stdout=self._log, stderr=self._log,
                                      pass_fds=(w,), env=env, cwd=str(root),
                                      start_new_session=True)
        os.close(w)
        self._rfd = r
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="stream-asr-reader")
        self._reader.start()
        if wait and not self._ready_event.wait(float(self.conf.get("stream_load_timeout_s", 180))):
            raise RuntimeError("stream ASR worker ready timeout")
        self._ready = self._ready_event.is_set()
        return self._ready

    def ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    def close(self):
        self._ready = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=8)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._rfd >= 0:
            try:
                os.close(self._rfd)
            except OSError:
                pass
            self._rfd = -1
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = os.read(self._rfd, n - len(buf))
            if not chunk:
                raise EOFError("stream ASR worker closed")
            buf.extend(chunk)
        return bytes(buf)

    def _read_loop(self):
        try:
            while True:
                typ = self._read_exact(1)
                (length,) = struct.unpack("<I", self._read_exact(4))
                payload = self._read_exact(length) if length else b""
                if typ == b"R":
                    self._ready_event.set()
                elif typ == b"T":
                    if len(payload) < 8:
                        continue
                    flags, _coverage = struct.unpack("<II", payload[:8])
                    text = payload[8:].decode("utf-8", "replace")
                    if flags & 1:
                        self._partial = text
                        if self.on_partial is not None:
                            try:
                                self.on_partial(text)
                            except Exception:
                                pass
                    if flags & 2:
                        self._final_text = text
                        self._final_event.set()
                elif typ == b"E":
                    print("[stream-asr] error:", payload.decode("utf-8", "replace"), flush=True)
                    break
        except Exception:
            pass

    def feed(self, pcm_bytes: bytes):
        if not pcm_bytes or not self._ready:
            return
        with self._lock:
            try:
                self._proc.stdin.write(struct.pack("<I", len(pcm_bytes)))
                self._proc.stdin.write(pcm_bytes)
                self._proc.stdin.flush()
            except Exception:
                pass

    def flush(self, timeout: float = 20.0) -> str:
        if not self._ready:
            return ""
        with self._lock:
            self._final_event.clear()
            try:
                self._proc.stdin.write(struct.pack("<I", 0))
                self._proc.stdin.flush()
            except Exception:
                return ""
        self._final_event.wait(timeout)
        text = self._final_text
        self._partial = ""
        return text
