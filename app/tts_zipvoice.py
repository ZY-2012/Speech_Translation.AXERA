"""Resident ZipVoice distill TTS worker client.

The C++ worker loads the NPU models once, accepts ``uid<TAB>text`` requests
(or pre-tokenised text through its Python tokenizer daemon) and streams
24 kHz S16LE PCM back over a dedicated response pipe.
"""
from __future__ import annotations

import os
import struct
import subprocess
import threading
import time
from pathlib import Path

SR = 24000
FRAME = 1920  # 80 ms


class ZipVoiceTts:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conf = cfg["tts"]
        self.enabled = bool(self.conf.get("enabled", True))
        self._proc = None
        self._rfd = -1
        self._log = None
        self._ready = False
        self._uid = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def load(self):
        if not self.enabled or self._ready:
            return
        root = Path(self.cfg["runtime"]["work_dir"]).parent
        log_dir = Path(self.cfg["runtime"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(self.conf.get("tmp_dir") or (root / ".work/zipvoice-tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        r, w = os.pipe()
        self._log = open(log_dir / "zipvoice.log", "ab", buffering=0)
        env = dict(os.environ)
        py = str(root / "runtime/python/bin")
        env["PATH"] = py + ":" + env.get("PATH", "")
        env["PYTHONPATH"] = str(root / "runtime/pylibs") + ":" + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = (str(root / "runtime/pylibs") +
                                  ":/soc/lib:/opt/lib:" + env.get("LD_LIBRARY_PATH", ""))
        args = [
            str(root / self.conf["binary"]),
            "--model-dir", str(root / self.conf["model_dir"]),
            "--prompt-wav", str(root / self.conf["prompt_wav"]),
            "--prompt-text", self.conf["prompt_text"],
            "--vocoder-model", str(root / self.conf["vocoder_model"]),
            "--repo-dir", str(root / self.conf["repo_dir"]),
            "--tmp-dir", str(tmp_dir),
            "--response-fd", str(w),
        ]
        if int(self.conf.get("num_step", 0)) > 0:
            args += ["--num-step", str(int(self.conf["num_step"]))]
        if float(self.conf.get("speed", 1.0)) != 1.0:
            args += ["--speed", str(self.conf["speed"])]
        if float(self.conf.get("guidance_scale", 0.0)) != 0.0:
            args += ["--guidance-scale", str(self.conf["guidance_scale"])]
        self._log.write(("\n=== start " + " ".join(args) + " ===\n").encode())
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                      stdout=self._log, stderr=self._log,
                                      pass_fds=(w,), env=env, cwd=str(root),
                                      start_new_session=True)
        os.close(w)
        self._rfd = r
        t0 = time.monotonic()
        uid, samples = self._read_header()
        if uid != 0 or samples != 0:
            raise RuntimeError("bad ZipVoice ready frame")
        self._ready = True
        print(f"[tts-zip] ready in {time.monotonic() - t0:.1f}s", flush=True)

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

    def warmup(self):
        # The first request also warms piper_phonemize/espeak data and the
        # first NPU decoder run; keep it out of the user-visible path.
        text = self.conf.get("warmup_text", "Hello, this is a test.")
        try:
            for _ in self.stream(text):
                pass
        except Exception:
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------ io
    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = os.read(self._rfd, n - len(buf))
            if not chunk:
                raise EOFError("ZipVoice worker closed")
            buf.extend(chunk)
        return bytes(buf)

    def _read_header(self):
        return struct.unpack("<II", self._read_exact(8))

    def stream(self, text: str, cancelled=None):
        if not self.enabled:
            return
        if not self._ready:
            self.load()
        text = (text or "").strip().replace("\t", " ").replace("\n", " ")
        if not text:
            return
        with self._lock:
            self._uid += 1
            uid = self._uid
            line = f"{uid}\t{text}\n"
            try:
                self._proc.stdin.write(line.encode("utf-8"))
                self._proc.stdin.flush()
            except Exception:
                import traceback
                traceback.print_exc()
                return
            while True:
                frame_uid, samples = self._read_header()
                if samples == 0:
                    break
                raw = self._read_exact(samples * 2)
                if frame_uid != uid:
                    continue
                if cancelled is not None and cancelled():
                    # Drain the rest of this request, keeping the worker in
                    # sync, then stop producing new clauses.
                    while True:
                        _, s = self._read_header()
                        if s == 0:
                            break
                        self._read_exact(s * 2)
                    return
                for i in range(0, samples, FRAME):
                    yield raw[i * 2:min(samples, i + FRAME) * 2]
