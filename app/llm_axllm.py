"""Qwen3-0.6B translation through the resident `axllm` OpenAI server."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class LlmError(RuntimeError):
    pass


class AxLlmTranslator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conf = cfg["llm"]
        self.base = f"http://127.0.0.1:{int(self.conf.get('port', 8000))}"
        self._lock = threading.Lock()
        self._proc = None
        self._log = None
        self._started = False

    # ------------------------------------------------------------------ setup
    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=1.0) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self, wait: bool = True) -> bool:
        if not self.conf.get("enabled", True):
            return False
        if self._health():
            self._started = True
            print("[llm] using existing axllm server", flush=True)
            return True
        binary = self.conf["binary"]
        model = self.conf["model_dir"]
        log_dir = Path(self.cfg["runtime"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(log_dir / "axllm.log", "ab", buffering=0)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/soc/lib:/opt/lib:" + env.get("LD_LIBRARY_PATH", "")
        args = [binary, "serve", model, "--port", str(self.conf.get("port", 8000))]
        self._log.write(("\n=== start %s ===\n" % " ".join(args)).encode())
        self._proc = subprocess.Popen(args, stdout=self._log, stderr=self._log,
                                      cwd=str(Path(self.cfg["runtime"]["work_dir"])),
                                      env=env, start_new_session=True)
        self._started = True
        if not wait:
            return True
        deadline = time.monotonic() + float(self.conf.get("load_timeout_s", 600))
        while time.monotonic() < deadline:
            if self._health():
                print("[llm] axllm server ready", flush=True)
                return True
            if self._proc.poll() is not None:
                raise LlmError("axllm server exited during load")
            time.sleep(1.0)
        raise LlmError("axllm server ready timeout")

    def ready(self) -> bool:
        return self._started and self._health()

    def close(self):
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
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None

    # ------------------------------------------------------------------ infer
    @staticmethod
    def detect_target(text: str, direction: str = "auto") -> str:
        if direction in ("en", "zh"):
            return direction
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        return "en" if cjk >= max(1, letters // 3) else "zh"

    def _prompt(self, text: str, target: str, context=None) -> str:
        if target == "en":
            base = ("Translate the following Chinese clause into natural, concise spoken "
                    "English. Output only the English translation of the new clause; do "
                    "not repeat or explain previous context.")
        else:
            base = ("把下面这句英文小句翻译成自然、简洁、口语化的中文。只输出新小句的中文译文，"
                    "不要重复前文，不要解释。")
        if context:
            lines = []
            for src, dst in context[-2:]:
                lines.append(f"S: {src}\nT: {dst}")
            base += "\n\nPrevious context (reference only):\n" + "\n".join(lines)
        return base + "\n\nNew clause:\n" + text

    def translate(self, text: str, target: str, on_token=None, context=None) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if not self.conf.get("enabled", True):
            return text
        with self._lock:
            if not self._started:
                self.start(wait=True)
            payload = {
                "model": self.conf.get("model_name", "AXERA-TECH/Qwen3-0.6B-GPTQ-Int4"),
                "messages": [
                    {"role": "system", "content": self.conf.get("system_prompt", "")},
                    {"role": "user", "content": self._prompt(text, target, context)},
                ],
                "max_tokens": int(self.conf.get("max_tokens", 80)),
                "temperature": float(self.conf.get("temperature", 0.0)),
                "top_p": float(self.conf.get("top_p", 1.0)),
                "stream": True,
                "enable_thinking": bool(self.conf.get("enable_thinking", False)),
            }
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            pieces: list[str] = []
            deadline = time.monotonic() + float(self.conf.get("timeout_s", 30))
            try:
                with urllib.request.urlopen(req, timeout=float(self.conf.get("timeout_s", 30))) as resp:
                    for raw in resp:
                        if time.monotonic() > deadline:
                            raise LlmError("translation timeout")
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except Exception:
                            continue
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        chunk = delta.get("content") or ""
                        # reasoning_content is intentionally ignored; we request
                        # enable_thinking=false so the content stream is the answer.
                        if chunk:
                            pieces.append(chunk)
                            if on_token is not None:
                                on_token(chunk)
            except urllib.error.URLError as exc:
                raise LlmError(f"axllm request failed: {exc}") from exc
            return self._clean("".join(pieces))

    @staticmethod
    def _clean(text: str) -> str:
        text = (text or "").strip()
        for prefix in ("English:", "English：", "中文：", "中文:", "Translation:",
                       "译文：", "译文:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        if len(text) >= 2 and text[0] in "\"“" and text[-1] in "\"”":
            text = text[1:-1].strip()
        return text
