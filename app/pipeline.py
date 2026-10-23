"""Non-streaming ASR -> clause MT -> ZipVoice session.

Pipeline:
  mic PCM -> energy VAD -> 0.6..3 s clause
          -> offline SenseVoice (punctuated text)
          -> source punctuation split
          -> Qwen3-0.6B clause MT (rolling context)
          -> target clause split
          -> resident ZipVoice distill C++ worker -> 24 kHz PCM
"""
from __future__ import annotations

import queue
import re
import threading
import time
import traceback

import numpy as np

from .asr_offline import OfflineAsr
from .clauses import ClauseSplitter, meaningful
from .llm_axllm import AxLlmTranslator
from .tts_service import TtsService
from .tts_zipvoice import FRAME as TTS_FRAME
from .tts_zipvoice import SR as TTS_SR
from .tts_zipvoice import ZipVoiceTts
from .vad_ten import Segmenter, make_vad

SRC_PUNCT = "。！？!?；;，,"


def split_source(text: str) -> list[str]:
    """Split punctuated ASR text into speakable clauses.

    The offline SenseVoice model emits Chinese punctuation.  Short fragments
    are merged so the MT model gets a semantically useful unit.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts: list[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in SRC_PUNCT:
            if meaningful(cur):
                parts.append(cur.strip())
            cur = ""
    if cur.strip() and meaningful(cur):
        parts.append(cur.strip())
    if not parts:
        parts = [text]

    merged: list[str] = []
    buf = ""
    for part in parts:
        buf = (buf + " " + part).strip() if buf else part
        # Keep very short fragments attached to the next clause.
        if len(re.sub(r"[\s，。！？；,.;!?]", "", buf)) >= 5:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = (merged[-1] + " " + buf).strip()
        else:
            merged.append(buf)

    # Hard cap for very long punctuation-free speech.
    out: list[str] = []
    for part in merged:
        while len(part) > 48:
            cut = max(part.rfind("，", 0, 48), part.rfind(" ", 0, 48))
            if cut < 8:
                cut = 48
            out.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        if part:
            out.append(part)
    return out


def _merge_sentence_text(a: str, b: str) -> str:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    if a[-1] in "。！？!?":
        a = a[:-1] + "，"
    return a + b


class Components:
    def __init__(self, cfg: dict, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self.asr = None if mock else OfflineAsr(cfg)
        self.llm = None if mock else AxLlmTranslator(cfg)
        self.tts = None if mock else ZipVoiceTts(cfg)

    def preload(self):
        if self.mock:
            return
        if self.llm is not None:
            try:
                self.llm.start(wait=True)
                print("[llm] qwen3-0.6b ready", flush=True)
                t0 = time.time()
                self.llm.translate("你好。", "en")
                print(f"[llm] warmup {time.time() - t0:.1f}s", flush=True)
            except Exception:
                traceback.print_exc()
        if self.asr is not None:
            try:
                self.asr.load()
                print("[asr] offline model ready", flush=True)
            except Exception:
                traceback.print_exc()
                self.asr = None
        if self.tts is not None:
            try:
                self.tts.load()
                t0 = time.time()
                self.tts.warmup()
                print(f"[tts] warmup {time.time() - t0:.1f}s", flush=True)
            except Exception:
                traceback.print_exc()
                self.tts = None

    def close(self):
        for obj in (self.asr, self.llm, self.tts):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


class Session:
    def __init__(self, cfg: dict, components: Components, send_event, send_audio,
                 mock: bool = False):
        self.cfg = cfg
        self.c = components
        self.send_event = send_event
        self.send_audio = send_audio
        self.mock = mock
        self.started = False
        self._t0 = time.monotonic()
        self._asr_q: queue.Queue = queue.Queue()
        self._mt_q: queue.Queue = queue.Queue()
        self._asr_thread = None
        self._mt_thread = None
        self._tts: TtsService | None = None
        self._seg_begin_at: dict[int, float] = {}
        self._seg_end_at: dict[int, float] = {}
        self._ctx: list[tuple[str, str]] = []
        self._tts_first_at = None
        self._tts_mute_until = 0.0
        self.echo_guard = True
        self._active_uid = 0
        self._fed_samples = 0
        self._mock_idx = 0
        self._mock_texts = [
            "今天天气很好，我们去公园散步吧。",
            "这个方案看起来不错，但是还需要再讨论一下细节。",
            "会议将在下午三点开始，请大家准时参加。",
        ]
        self._vad = make_vad(cfg, mock=mock)
        self.segmenter = Segmenter(
            cfg, self._vad,
            on_start=self._on_speech_start,
            on_end=self._on_speech_end)

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, **payload):
        payload["type"] = kind
        payload["t"] = round(time.monotonic() - self._t0, 3)
        self.send_event(payload)

    def _tts_emit(self, kind: str, **payload):
        uid = payload.get("uid")
        now = time.monotonic()
        if uid in self._seg_begin_at:
            payload["since_speech_start_ms"] = round(
                (now - self._seg_begin_at[uid]) * 1000, 1)
        if kind == "tts_first_frame" and uid in self._seg_end_at:
            payload["since_speech_end_ms"] = round(
                (now - self._seg_end_at[uid]) * 1000, 1)
        if kind == "tts_first_frame":
            self._tts_first_at = now
        if kind == "tts_end":
            frames = int(payload.get("frames", 0))
            base = self._tts_first_at or now
            self._tts_mute_until = max(
                self._tts_mute_until, base + frames * 0.08 + 0.25)
            self._tts_first_at = None
        self._emit(kind, **payload)

    # ------------------------------------------------------------------ control
    def start(self):
        self.started = True
        self._t0 = time.monotonic()

        def synth(text, cancelled):
            if self.c.tts is None:
                yield from self._mock_tts(text)
            else:
                yield from self.c.tts.stream(text, cancelled=cancelled)

        self._tts = TtsService(synth_fn=synth, send_audio=self.send_audio,
                               emit=self._tts_emit)
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True,
                                            name="asr-offline")
        self._mt_thread = threading.Thread(target=self._mt_loop, daemon=True,
                                           name="mt-clause")
        self._asr_thread.start()
        self._mt_thread.start()
        self._emit("ready", mock=self.mock)

    def cancel_tts(self):
        if self._tts is not None:
            self._tts.cancel()
        self._tts_mute_until = 0.0
        self._emit("tts_cancel")

    def stop(self):
        if not self.started:
            return
        self.started = False
        self.segmenter.flush()
        self._asr_q.put(None)
        if self._asr_thread is not None:
            self._asr_thread.join(timeout=15)
            self._asr_thread = None
        self._mt_q.put(None)
        if self._mt_thread is not None:
            self._mt_thread.join(timeout=30)
            self._mt_thread = None
        if self._tts is not None:
            self._tts.close()
            self._tts = None
        try:
            self._vad.close()
        except Exception:
            pass
        self._emit("stopped")

    # ------------------------------------------------------------------ audio
    def feed(self, pcm_bytes: bytes):
        if not self.started or not pcm_bytes:
            return
        pcm = np.frombuffer(pcm_bytes, dtype="<i2")
        if pcm.size:
            if self.echo_guard and time.monotonic() < self._tts_mute_until:
                pcm = np.zeros_like(pcm)
            self._fed_samples += pcm.size
            self.segmenter.push(pcm)

    def _on_speech_start(self, index: int, start_sample: int, pre_roll: np.ndarray):
        self._active_uid = index
        self._seg_begin_at[index] = time.monotonic()
        self._seg_end_at.pop(index, None)
        if self.cfg["runtime"].get("barge_in", False):
            if self._tts is not None and self._tts.active:
                self.cancel_tts()
        self._emit("speech_start", segment=index,
                   audio_ms=round(pre_roll.size * 1000 /
                                  self.cfg["audio"]["sample_rate"], 1))

    def _on_speech_end(self, segment: np.ndarray, start_sample: int,
                       end_sample: int, seconds: float):
        uid = self._active_uid
        self._active_uid = 0
        self._seg_end_at[uid] = time.monotonic()
        self._emit("speech_end", segment=uid, seconds=round(seconds, 2),
                   samples=int(segment.size))
        min_s = float(self.cfg["audio"].get("min_speech_ms", 600)) / 1000.0
        if seconds < min_s or segment.size < int(min_s * self.cfg["audio"]["sample_rate"]):
            self._emit("speech_skipped", segment=uid, seconds=round(seconds, 2),
                       reason="too_short")
            return
        self._asr_q.put({"uid": uid, "pcm": segment.tobytes(),
                         "audio_s": seconds, "end_at": self._seg_end_at[uid]})

    # ------------------------------------------------------------------ asr
    def _asr_loop(self):
        merge_s = float(self.cfg["audio"].get("sentence_merge_ms", 600)) / 1000.0
        pending: dict | None = None

        def flush_pending():
            nonlocal pending
            if pending is None:
                return
            p = pending
            pending = None
            latency = (time.monotonic() - p["first_end"]) * 1000
            asr_ms = p.get("asr_ms", 0.0)
            self._emit("asr_final", uid=p["uid"], text=p["text"],
                       audio_s=round(p["audio_s"], 2),
                       latency_ms=round(latency, 1),
                       asr_ms=round(asr_ms, 1),
                       merge_ms=round(max(0.0, latency - asr_ms), 1))
            self._mt_q.put({"uid": p["uid"], "text": p["text"],
                            "audio_s": p["audio_s"]})

        while True:
            timeout = merge_s if pending is not None else None
            try:
                job = self._asr_q.get(timeout=timeout)
            except queue.Empty:
                flush_pending()
                continue
            if job is None:
                flush_pending()
                break
            try:
                if self.mock:
                    text = self._mock_texts[self._mock_idx % len(self._mock_texts)]
                    self._mock_idx += 1
                else:
                    text = self.c.asr.run(job["pcm"]) if self.c.asr is not None else ""
                text = (text or "").strip()
                if not text:
                    self._emit("asr_empty", uid=job["uid"])
                    continue
                now = time.monotonic()
                if pending is not None and now - pending["last_at"] <= merge_s:
                    pending["text"] = _merge_sentence_text(pending["text"], text)
                    pending["audio_s"] += job["audio_s"]
                    pending["asr_ms"] += (now - job["end_at"]) * 1000
                    pending["last_at"] = now
                else:
                    flush_pending()
                    pending = {"uid": job["uid"], "text": text,
                               "audio_s": job["audio_s"],
                               "first_end": job["end_at"], "last_at": now,
                               "asr_ms": (now - job["end_at"]) * 1000}
            except Exception:
                traceback.print_exc()
                self._emit("error", message="ASR failed")
        self._emit("asr_worker_stopped")

    # ------------------------------------------------------------------ mt/tts
    def _submit_clause(self, uid: int, text: str) -> bool:
        if self._tts is None or not self.cfg["tts"].get("enabled", True):
            return False
        if not meaningful(text):
            return False
        return self._tts.submit(text, uid, last=False)

    def _mt_loop(self):
        while True:
            job = self._mt_q.get()
            if job is None:
                break
            try:
                self._translate(job["uid"], job["text"])
            except Exception:
                traceback.print_exc()
                self._emit("error", message="translation failed")
        self._emit("mt_worker_stopped")

    def _translate(self, uid: int, source: str):
        source = (source or "").strip()
        if not source:
            return
        cfg = self.cfg
        target = AxLlmTranslator.detect_target(source, cfg["llm"].get("direction", "auto"))
        self._emit("translate_start", target=target, source=source, uid=uid)
        clauses = split_source(source)
        all_text: list[str] = []
        first_token_at = None
        first_clause_at = None
        submitted = 0
        t0 = time.monotonic()

        for clause in clauses:
            splitter = ClauseSplitter(
                min_chars=int(cfg["tts"].get("clause_min_chars", 6)),
                max_chars=int(cfg["tts"].get("clause_max_chars", 20)))

            def submit_target(chunks):
                nonlocal first_clause_at, submitted
                for piece in chunks:
                    if self._submit_clause(uid, piece):
                        submitted += 1
                        if first_clause_at is None:
                            first_clause_at = time.monotonic()

            def on_token(chunk: str):
                nonlocal first_token_at
                if first_token_at is None:
                    first_token_at = time.monotonic()
                self._emit("trans_token", text=chunk, target=target, uid=uid)
                submit_target(splitter.feed(chunk))

            translation = ""
            if self.mock:
                translation = "This is a mock translation for the browser test."
                for ch in translation:
                    on_token(ch)
                    time.sleep(0.006)
            elif self.c.llm is not None:
                context = self._ctx[-2:] if cfg["llm"].get("use_context", False) else None
                translation = self.c.llm.translate(
                    clause, target, on_token, context=context)
            if not translation:
                translation = clause
                on_token(translation)
            submit_target([splitter.flush()])
            all_text.append(translation.strip())
            self._ctx.append((clause, translation.strip()))
            self._ctx = self._ctx[-6:]

        final_text = " ".join(t for t in all_text if t).strip()
        if submitted and self._tts is not None:
            self._tts.submit("", uid, last=True)
        self._emit("trans_final", text=final_text, target=target, uid=uid,
                   ms=round((time.monotonic() - t0) * 1000, 1),
                   ttft_ms=(round((first_token_at - t0) * 1000, 1)
                            if first_token_at else None),
                   first_clause_ms=(round((first_clause_at - t0) * 1000, 1)
                                    if first_clause_at else None),
                   clauses=submitted)
        print(f"[mt] uid={uid} src_chars={len(source)} clauses={len(clauses)} "
              f"target_chars={len(final_text)} "
              f"first_clause={((first_clause_at - t0) * 1000 if first_clause_at else -1):.0f}ms "
              f"total={(time.monotonic() - t0) * 1000:.0f}ms", flush=True)

    def _mock_tts(self, text: str):
        total = max(1.0, min(4.0, len(text) / 8.0))
        n = int(TTS_SR * total)
        t = np.arange(n, dtype=np.float32) / TTS_SR
        freq = 180 + 40 * np.sin(2 * np.pi * 0.7 * t)
        wave = 0.08 * np.sin(2 * np.pi * freq * t) * np.hanning(n)
        pcm = (wave * 32767).astype(np.int16)
        for i in range(0, n, TTS_FRAME):
            yield pcm[i:i + TTS_FRAME].tobytes()
