"""Resident FireRedPunc AX650 worker client.

The main sidecar runs on the portable Python 3.11 runtime, which does not
ship transformers.  The Punc worker therefore runs under the system Python
3.10 (which has transformers + axengine) and speaks a tiny line protocol:

  stdin:  uid<TAB>text\n
  fd:     READY\n
          uid<TAB>punctuated-text\n
"""
from __future__ import annotations

import os
import re
import select
import subprocess
import threading
from pathlib import Path

SENT_PUNCT = re.compile(r"[。！？!?]")


class PuncClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conf = cfg.get("punc", {})
        self.enabled = bool(self.conf.get("enabled", True))
        self.mode = str(self.conf.get("mode", "auto")).lower()
        self._proc = None
        self._rfd = None
        self._log = None
        self._ready = False
        self._uid = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def start(self, wait: bool = True) -> bool:
        if not self.enabled or self.mode == "off":
            return False
        if self._ready:
            return True
        root = Path(self.cfg["runtime"]["work_dir"]).parent
        log_dir = Path(self.cfg["runtime"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        worker = Path(self.conf.get("worker", "app/punc_worker.py"))
        runtime_dir = Path(self.conf.get("runtime_dir", "../FireRedPunc.AXERA/python"))
        model_dir = Path(self.conf.get("model_dir", "../FireRedPunc.AXERA/models"))
        if not worker.is_absolute():
            worker = root / worker
        if not runtime_dir.is_absolute():
            runtime_dir = root / runtime_dir
        if not model_dir.is_absolute():
            model_dir = root / model_dir

        py = str(self.conf.get("python", "/usr/bin/python3"))
        r, w = os.pipe()
        self._log = open(log_dir / "punc.log", "ab", buffering=0)
        env = dict(os.environ)
        # The worker runs under the system Python 3.10.  Do not inherit the
        # main process's Python 3.11 PYTHONPATH/pylibs.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["LD_LIBRARY_PATH"] = "/soc/lib:/opt/lib"
        args = [
            py, str(worker),
            "--model-dir", str(model_dir),
            "--runtime-dir", str(runtime_dir),
            "--response-fd", str(w),
        ]
        self._log.write(("\n=== start " + " ".join(args) + " ===\n").encode())
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                      stdout=self._log, stderr=self._log,
                                      pass_fds=(w,), env=env, cwd=str(root),
                                      start_new_session=True)
        os.close(w)
        self._rfd = os.fdopen(r, "r", encoding="utf-8", buffering=1)
        if wait:
            timeout = float(self.conf.get("start_timeout_s", 120))
            ready, _, _ = select.select([self._rfd], [], [], timeout)
            if not ready:
                raise RuntimeError(f"punc worker start timeout ({timeout:.0f}s)")
            line = self._rfd.readline().strip()
            if line != "READY":
                raise RuntimeError(f"punc worker not ready: {line!r}")
        self._ready = True
        return True

    def ready(self) -> bool:
        return (self._ready and self._proc is not None
                and self._proc.poll() is None)

    def close(self):
        self._ready = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._rfd is not None:
            try:
                self._rfd.close()
            except Exception:
                pass
            self._rfd = None
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None

    # ------------------------------------------------------------------ infer
    def apply(self, text: str) -> str:
        text = (text or "").strip()
        if not text or not self.enabled or self.mode == "off":
            return text
        if self.mode == "auto" and SENT_PUNCT.search(text):
            return text
        if len(re.sub(r"\s+", "", text)) < 2:
            return text
        if not self.ready():
            return text
        text = text.replace("\t", " ").replace("\n", " ").strip()
        with self._lock:
            self._uid += 1
            uid = self._uid
            try:
                self._proc.stdin.write(f"{uid}\t{text}\n".encode("utf-8"))
                self._proc.stdin.flush()
                timeout = float(self.conf.get("request_timeout_s", 5))
                ready, _, _ = select.select([self._rfd], [], [], timeout)
                if not ready:
                    raise TimeoutError(f"punc request timeout ({timeout:.1f}s)")
                line = self._rfd.readline()
                if not line:
                    raise EOFError("punc worker closed")
                got, out = line.rstrip("\n").split("\t", 1)
                if int(got) != uid:
                    raise RuntimeError(f"punc uid mismatch {got} != {uid}")
                return out.strip()
            except Exception as exc:
                print(f"[punc] fallback: {exc}", flush=True)
                return text
