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
from pathlib import Path

import numpy as np

from .asr_offline import OfflineAsr
from .asr_stream import StreamAsr
from .clauses import ClauseSplitter, meaningful
from .llm_axllm import AxLlmTranslator
from .number_text import normalize_numbers
from .opus_mt import OpusMtTranslator
from .tts_service import TtsService
from .tts_zipvoice import FRAME as TTS_FRAME
from .tts_zipvoice import SR as TTS_SR
from .tts_zipvoice import ZipVoiceTts
from .vad_ten import Segmenter, make_vad

SRC_PUNCT = "。！？!?；;，,"
_FILLER_CHARS = set("嗯啊呃哦唉诶额唔哈呀")


def _filler_only(clean: str) -> bool:
    return bool(clean) and all(ch in _FILLER_CHARS for ch in clean)


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


class Components:
    def __init__(self, cfg: dict, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self.asr = None if mock else OfflineAsr(cfg)
        self.asr_stream = None
        if not mock and cfg["asr"].get("stream_enabled", False):
            self.asr_stream = StreamAsr(cfg)
        self.tts = None if mock else ZipVoiceTts(cfg)
        self.tts_zh = None
        if not mock and cfg["tts"].get("zh_prompt_wav"):
            # A separate resident worker keeps the Chinese reference timbre
            # for EN->ZH while the default worker handles ZH->EN.
            cfg_zh = dict(cfg)
            cfg_zh["tts"] = dict(cfg["tts"])
            cfg_zh["tts"]["prompt_wav"] = cfg["tts"]["zh_prompt_wav"]
            cfg_zh["tts"]["prompt_text"] = cfg["tts"].get("zh_prompt_text", "")
            cfg_zh["tts"]["warmup_text"] = cfg["tts"].get(
                "zh_warmup_text", "你好，这是一个测试。")
            cfg_zh["tts"]["tmp_dir"] = cfg["tts"].get(
                "zh_tmp_dir", ".work/zipvoice-tmp-zh")
            self.tts_zh = ZipVoiceTts(cfg_zh)
        self.tts_custom = None
        self.tts_custom_loading = False
        self.tts_custom_error = ""
        self.tts_custom_name = ""
        self._voice_lock = threading.Lock()
        self.asr_mode = cfg["asr"].get("mode", "hybrid")
        self.tts_enabled = bool(cfg["tts"].get("enabled", True))
        self.tts_continuous = bool(cfg["tts"].get("continuous", True))
        self.tts_voice = cfg["tts"].get("voice", "auto")
        self.llm_qwen = None
        self.llm_opus = None
        self.llm = None
        if not mock:
            if cfg["llm"].get("qwen_enabled", True):
                self.llm_qwen = AxLlmTranslator(cfg)
            if cfg["llm"].get("opus_enabled", True):
                self.llm_opus = OpusMtTranslator(cfg)
            self.set_mt_backend(cfg["llm"].get("backend", "opus"))

    def set_mt_backend(self, backend: str) -> str:
        backend = backend if backend in ("qwen", "opus", "none") else "opus"
        if backend == "none":
            self.mt_backend = "none"
            self.llm = None
            return self.mt_backend
        obj = self.llm_qwen if backend == "qwen" else self.llm_opus
        if obj is None:
            obj = self.llm_opus if backend == "qwen" else self.llm_qwen
            backend = "opus" if obj is self.llm_opus else "qwen"
        self.mt_backend = backend
        self.llm = obj
        return self.mt_backend

    def set_asr_mode(self, mode: str) -> str:
        self.asr_mode = "offline" if mode == "offline" else "hybrid"
        return self.asr_mode

    def set_tts_enabled(self, enabled: bool):
        self.tts_enabled = bool(enabled)

    def set_tts_continuous(self, enabled: bool):
        self.tts_continuous = bool(enabled)

    def set_tts_voice(self, voice: str) -> str:
        if voice == "custom" and self.tts_custom is None:
            voice = "auto"
        self.tts_voice = voice if voice in ("en", "zh", "custom") else "auto"
        return self.tts_voice

    def tts_for(self, target: str):
        if self.tts_voice == "custom" and self.tts_custom is not None:
            return self.tts_custom
        if self.tts_voice == "zh" and self.tts_zh is not None:
            return self.tts_zh
        if self.tts_voice == "en" and self.tts is not None:
            return self.tts
        if target == "zh" and self.tts_zh is not None:
            return self.tts_zh
        return self.tts or self.tts_zh

    def load_custom_voice(self, wav_path: Path, prompt_text: str,
                          name: str = "") -> dict:
        with self._voice_lock:
            if self.tts_custom_loading:
                return {"ok": False, "error": "voice is already loading"}
            self.tts_custom_loading = True
            self.tts_custom_error = ""
            self.tts_custom_name = name or wav_path.name
        th = threading.Thread(target=self._load_custom_voice,
                              args=(wav_path, prompt_text),
                              daemon=True, name="tts-custom-load")
        th.start()
        return {"ok": True, "loading": True, "name": self.tts_custom_name}

    def _load_custom_voice(self, wav_path: Path, prompt_text: str):
        cfg = dict(self.cfg)
        cfg["tts"] = dict(self.cfg["tts"])
        cfg["tts"]["prompt_wav"] = str(wav_path)
        cfg["tts"]["prompt_text"] = prompt_text
        cfg["tts"]["tmp_dir"] = ".work/zipvoice-tmp-custom"
        cfg["tts"]["warmup_text"] = (
            "你好，这是一个测试。"
            if re.search(r"[\u4e00-\u9fff]", prompt_text)
            else "Hello, this is a test.")
        engine = None
        try:
            engine = ZipVoiceTts(cfg)
            engine.load()
            t0 = time.time()
            engine.warmup()
            print(f"[tts-custom] warmup {time.time() - t0:.1f}s", flush=True)
        except Exception:
            traceback.print_exc()
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass
            with self._voice_lock:
                self.tts_custom_loading = False
                self.tts_custom_error = "reference voice load failed"
            return
        old = None
        with self._voice_lock:
            old = self.tts_custom
            self.tts_custom = engine
            self.tts_custom_loading = False
            self.tts_custom_error = ""
            self.tts_voice = "custom"
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        print(f"[tts-custom] ready name={self.tts_custom_name!r}",
              flush=True)

    def preload(self):
        if self.mock:
            return
        for name, obj in (("qwen", self.llm_qwen), ("opus", self.llm_opus)):
            if obj is None:
                continue
            try:
                obj.start(wait=True)
                print(f"[llm] {name} ready", flush=True)
                t0 = time.time()
                obj.translate("你好。", "en")
                print(f"[llm] {name} warmup {time.time() - t0:.1f}s", flush=True)
            except Exception:
                traceback.print_exc()
                if name == "qwen":
                    self.llm_qwen = None
                else:
                    self.llm_opus = None
        self.set_mt_backend(self.mt_backend)
        if self.asr_stream is not None:
            try:
                self.asr_stream.start(wait=True)
                print("[asr] streaming model ready", flush=True)
            except Exception:
                traceback.print_exc()
                self.asr_stream = None
        if self.asr is not None:
            try:
                self.asr.load()
                print("[asr] offline model ready", flush=True)
            except Exception:
                traceback.print_exc()
                self.asr = None
        for name, obj in (("en", self.tts), ("zh", self.tts_zh)):
            if obj is None:
                continue
            try:
                obj.load()
                t0 = time.time()
                obj.warmup()
                print(f"[tts-{name}] warmup {time.time() - t0:.1f}s",
                      flush=True)
            except Exception:
                traceback.print_exc()
                if name == "zh":
                    self.tts_zh = None
                else:
                    self.tts = None

    def close(self):
        seen = set()
        for obj in (self.asr_stream, self.asr, self.llm_qwen, self.llm_opus,
                    self.tts, self.tts_zh, self.tts_custom):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            try:
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
        self.echo_guard = False
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

        def synth(text, cancelled, target):
            engine = self.c.tts_for(target)
            if engine is None:
                yield from self._mock_tts(text)
            else:
                yield from engine.stream(text, cancelled=cancelled)

        if self.c.asr_stream is not None:
            self.c.asr_stream.on_partial = self._on_asr_partial
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
            if self.c.asr_stream is not None and self.c.asr_mode != "offline":
                self.c.asr_stream.feed(pcm.tobytes())
            self._fed_samples += pcm.size
            self.segmenter.push(pcm)

    def _on_asr_partial(self, text: str):
        if not self.started or not meaningful(text):
            return
        # The stream worker keeps running on room noise.  Only show partials
        # once VAD has accumulated real speech; otherwise "嗯"/"我" flashes
        # before the user starts talking.
        if not self.segmenter.in_speech or self.segmenter.speech_ms < 600:
            return
        clean = re.sub(r"[\s\W_]", "", text or "")
        if len(clean) < 3 or _filler_only(clean):
            return
        self._emit("asr_partial", text=text or "")

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
        # SenseVoice can hallucinate full sentences from very low-level room
        # noise.  Reject segments whose RMS is below the configured floor.
        if segment.size:
            x = segment.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x)))
        else:
            rms = 0.0
        min_rms = float(self.cfg["audio"].get("min_rms", 0.003))
        if rms < min_rms:
            self._emit("speech_skipped", segment=uid, seconds=round(seconds, 2),
                       reason="low_level")
            return
        self._asr_q.put({"uid": uid, "pcm": segment.tobytes(),
                         "audio_s": seconds, "end_at": self._seg_end_at[uid]})

    # ------------------------------------------------------------------ asr
    def _asr_loop(self):
        """Finalize each VAD utterance with streaming + offline SenseVoice.

        The resident stream worker keeps cross-fragment context and emits
        partials while speech is ongoing.  At the endpoint, flush the stream
        and run AX_ASR_RunPCM on the full utterance; prefer the offline text
        for punctuation/homophones, falling back to the stream only if the
        offline text looks truncated.  ``sentence_merge_ms`` > 0 still merges
        adjacent PCM fragments first.
        """
        merge_s = float(self.cfg["audio"].get("sentence_merge_ms", 800)) / 1000.0
        pending: dict | None = None

        def flush_pending():
            nonlocal pending
            if pending is None:
                return
            p = pending
            pending = None
            pcm = b"".join(p["pcm"])
            t_asr = time.monotonic()
            stream_text = ""
            offline_text = ""
            try:
                if self.mock:
                    text = self._mock_texts[self._mock_idx % len(self._mock_texts)]
                    self._mock_idx += 1
                else:
                    if self.c.asr_stream is not None and self.c.asr_mode != "offline":
                        stream_text = self.c.asr_stream.flush()
                    if self.c.asr is not None:
                        offline_text = self.c.asr.run(pcm)
                    # Streaming SenseVoice keeps cross-fragment context; the
                    # offline segment is used when it is complete enough to
                    # supply punctuation/better homophones.
                    sk = lambda x: re.sub(r"[\s\W_]", "", x or "")
                    if offline_text and stream_text:
                        # Keep the previous "longer text wins" behaviour for
                        # unpunctuated output; additionally prefer the offline
                        # pass when it has punctuation and is not missing a
                        # large tail.  This keeps live recognition stable while
                        # still getting offline punctuation/refinement.
                        o_len, s_len = len(sk(offline_text)), len(sk(stream_text))
                        has_punct = bool(re.search(r"[。！？!?；;，,]", offline_text))
                        text = (offline_text if (o_len >= s_len or
                                                 (has_punct and o_len >= s_len * 0.75))
                                else stream_text)
                    else:
                        text = offline_text or stream_text
            except Exception:
                traceback.print_exc()
                self._emit("error", message="ASR failed")
                return
            asr_ms = (time.monotonic() - t_asr) * 1000
            text = (text or "").strip()
            # Drop noise and VAD fragments.  A complete Chinese utterance
            # normally gets terminal punctuation from the offline pass; a
            # short unpunctuated result is usually a clipped fragment or a
            # nearby speaker's partial sentence.
            clean = re.sub(r"[\s\W_]", "", text or "")
            min_chars = max(1, int(self.cfg["asr"].get("min_text_chars", 2)))
            drop_short = max(0, int(self.cfg["asr"].get(
                "drop_unpunctuated_chars", 12)))
            has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text or ""))
            terminal = bool(re.search(r"[。！？!?]", text or ""))
            if (not clean or len(clean) < min_chars or _filler_only(clean) or
                    (has_cjk and drop_short and len(clean) <= drop_short
                     and not terminal)):
                self._emit("asr_empty", uid=p["uid"])
                return
            # Keep Arabic numerals out of both the transcript and MT input.
            # Normalize with the source language inferred from the target.
            target_lang = AxLlmTranslator.detect_target(
                text, self.cfg["llm"].get("direction", "auto"))
            src_lang = "zh" if target_lang == "en" else "en"
            text = normalize_numbers(text, src_lang)
            total_ms = (time.monotonic() - p["first_end"]) * 1000
            self._emit("asr_final", uid=p["uid"], text=text,
                       audio_s=round(p["audio_s"], 2),
                       latency_ms=round(total_ms, 1),
                       asr_ms=round(asr_ms, 1),
                       merge_ms=round(max(0.0, total_ms - asr_ms), 1))
            self._mt_q.put({"uid": p["uid"], "text": text,
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
            now = time.monotonic()
            if pending is not None and now - pending["last_at"] <= merge_s:
                pending["pcm"].append(job["pcm"])
                pending["audio_s"] += job["audio_s"]
                pending["last_at"] = now
            else:
                flush_pending()
                pending = {"uid": job["uid"], "pcm": [job["pcm"]],
                           "audio_s": job["audio_s"],
                           "first_end": job["end_at"], "last_at": now}
        self._emit("asr_worker_stopped")

    # ------------------------------------------------------------------ mt/tts
    def _submit_clause(self, uid: int, text: str, target: str) -> bool:
        if self._tts is None or not self.c.tts_enabled:
            return False
        if not meaningful(text):
            return False
        return self._tts.submit(text, uid, last=False, target=target)

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
        # Continuous mode synthesises the complete translation in one
        # ZipVoice request.  This removes the ~1.4 s first-frame gap between
        # target clauses and is much smoother for long WAV/live utterances.
        continuous = bool(self.c.tts_continuous)
        self._emit("translate_start", target=target, source=source, uid=uid)
        clauses = split_source(source)
        all_text: list[str] = []
        first_token_at = None
        first_clause_at = None
        submitted = 0
        t0 = time.monotonic()

        for clause in clauses:
            splitter = None
            if not continuous:
                splitter = ClauseSplitter(
                    min_chars=int(cfg["tts"].get("clause_min_chars", 6)),
                    max_chars=int(cfg["tts"].get("clause_max_chars", 20)))

            def submit_target(chunks):
                nonlocal first_clause_at, submitted
                for piece in chunks:
                    # ZipVoice often drops Arabic digits; speak numbers as
                    # Chinese/English words before synthesis.
                    piece = normalize_numbers(piece, target)
                    if self._submit_clause(uid, piece, target):
                        submitted += 1
                        if first_clause_at is None:
                            first_clause_at = time.monotonic()

            def on_token(chunk: str):
                nonlocal first_token_at
                if first_token_at is None:
                    first_token_at = time.monotonic()
                self._emit("trans_token", text=chunk, target=target, uid=uid)
                if splitter is not None:
                    # Normalize before clause splitting so thousands commas
                    # (1,000) cannot be mistaken for target punctuation.
                    submit_target(splitter.feed(normalize_numbers(chunk, target)))

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
            else:
                translation = clause
                on_token(translation)
            if not translation:
                translation = clause
                on_token(translation)
            if splitter is not None:
                submit_target([splitter.flush()])
            all_text.append(translation.strip())
            self._ctx.append((clause, translation.strip()))
            self._ctx = self._ctx[-6:]

        final_text = normalize_numbers(
            " ".join(t for t in all_text if t).strip(), target)
        if continuous:
            if (self._tts is not None and self.c.tts_enabled
                    and meaningful(final_text)):
                if self._tts.submit(final_text, uid, last=True, target=target):
                    submitted = 1
                    first_clause_at = time.monotonic()
        elif submitted and self._tts is not None:
            self._tts.submit("", uid, last=True, target=target)
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
