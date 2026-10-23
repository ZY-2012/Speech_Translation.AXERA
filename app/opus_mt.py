"""Resident Opus-MT AX650 worker client.

Both ZH->EN and EN->ZH models stay loaded in the C++ worker.  A request is
one line ``uid<TAB>target<TAB>text`` and the reply is
``uint32 uid | uint32 n | n bytes UTF-8``.
"""
from __future__ import annotations

import os
import struct
import subprocess
import threading
import time
from pathlib import Path


class OpusMtTranslator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conf = cfg["llm"].get("opus", {})
        self._proc = None
        self._rfd = -1
        self._log = None
        self._ready = False
        self._uid = 0
        self._lock = threading.Lock()

    def start(self, wait: bool = True) -> bool:
        if not self.conf.get("enabled", True):
            return False
        if self._ready:
            return True
        root = Path(self.cfg["runtime"]["work_dir"]).parent
        log_dir = Path(self.cfg["runtime"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        r, w = os.pipe()
        self._log = open(log_dir / "opus_mt.log", "ab", buffering=0)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/soc/lib:/opt/lib:" + env.get("LD_LIBRARY_PATH", "")
        args = [
            str(root / self.conf["binary"]),
            "--model-root", str(root / self.conf["model_root"]),
            "--response-fd", str(w),
        ]
        self._log.write(("\n=== start " + " ".join(args) + " ===\n").encode())
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                      stdout=self._log, stderr=self._log,
                                      pass_fds=(w,), env=env, cwd=str(root),
                                      start_new_session=True)
        os.close(w)
        self._rfd = r
        uid, n = self._read_header()
        if uid != 0 or n != 0:
            raise RuntimeError("bad Opus-MT ready frame")
        self._ready = True
        print("[opus-mt] ready", flush=True)
        return True

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
                raise EOFError("Opus-MT worker closed")
            buf.extend(chunk)
        return bytes(buf)

    def _read_header(self):
        return struct.unpack("<II", self._read_exact(8))

    def translate(self, text: str, target: str, on_token=None, context=None) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if not self._ready:
            self.start(wait=True)
        target = "zh" if target == "zh" else "en"
        with self._lock:
            self._uid += 1
            uid = self._uid
            line = f"{uid}\t{target}\t{text.replace(chr(9), ' ').replace(chr(10), ' ')}\n"
            self._proc.stdin.write(line.encode("utf-8"))
            self._proc.stdin.flush()
            got_uid, n = self._read_header()
            if got_uid != uid:
                raise RuntimeError(f"Opus-MT uid mismatch {got_uid} != {uid}")
            raw = self._read_exact(n) if n else b""
            out = raw.decode("utf-8", "replace").strip()
        if on_token is not None and out:
            on_token(out)
        return out
