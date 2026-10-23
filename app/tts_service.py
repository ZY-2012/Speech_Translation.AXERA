"""Resident Pocket-TTS consumer: clauses are queued, synthesised and streamed."""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass


@dataclass
class TtsJob:
    text: str
    uid: int
    last: bool
    epoch: int
    t_submit: float


class TtsService:
    """One synthesis thread per session; the model itself is loaded once.

    synth_fn(text, cancelled) is a generator of 24 kHz S16LE byte chunks.
    Events emitted: tts_start / tts_first_frame / tts_end (with uid, frames,
    ms, cancelled).  The caller emits tts_cancel when it calls cancel().
    """

    def __init__(self, synth_fn, send_audio, emit, on_synth=None,
                 max_queue: int = 64):
        self.synth_fn = synth_fn
        self.send_audio = send_audio
        self.emit = emit
        self.on_synth = on_synth
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._epoch = 0
        self._closed = False
        self._lock = threading.Lock()
        self._started: set[int] = set()
        self._first_frame: set[int] = set()
        self._frames: dict[int, int] = {}
        self._synth_ms: dict[int, float] = {}
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="tts")
        self._thread.start()

    @property
    def active(self) -> bool:
        return bool(self._started)

    def submit(self, text: str, uid: int, last: bool = False) -> bool:
        if self._closed:
            return False
        job = TtsJob(text=text or "", uid=uid, last=last,
                     epoch=self._epoch, t_submit=time.monotonic())
        try:
            self._q.put(job, timeout=5.0)
            return True
        except queue.Full:
            print("[tts] queue full, dropping clause", flush=True)
            return False

    def cancel(self):
        with self._lock:
            self._epoch += 1

    def close(self):
        self._closed = True
        self._stop.set()
        self.cancel()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=8)

    # ------------------------------------------------------------------ loop
    def _begin(self, job: TtsJob):
        if job.uid not in self._started:
            self._started.add(job.uid)
            self._frames.setdefault(job.uid, 0)
            self._synth_ms.setdefault(job.uid, 0.0)
            self.emit("tts_start", uid=job.uid, text=job.text[:80])

    def _finish(self, uid: int, cancelled: bool):
        if uid not in self._started:
            return
        frames = self._frames.pop(uid, 0)
        synth = self._synth_ms.pop(uid, 0.0)
        self._started.discard(uid)
        self._first_frame.discard(uid)
        audio_ms = frames * 80
        rtf = (synth / audio_ms) if audio_ms else 0.0
        print(f"[tts] uid={uid} frames={frames} audio={audio_ms}ms "
              f"synth={synth:.1f}ms rtf={rtf:.2f} cancelled={cancelled}",
              flush=True)
        self.emit("tts_end", uid=uid, frames=frames,
                  ms=round(synth, 1), cancelled=cancelled)

    def _loop(self):
        while not self._stop.is_set():
            job = self._q.get()
            if job is None:
                break
            with self._lock:
                current_epoch = self._epoch
            if job.epoch != current_epoch:
                continue
            if not job.text.strip():
                if job.last:
                    self._finish(job.uid, cancelled=False)
                continue

            self._begin(job)
            t0 = time.monotonic()
            first = True
            frames = 0
            cancelled = False
            if self.on_synth is not None:
                self.on_synth(True)
            try:
                for pcm in self.synth_fn(
                        job.text,
                        lambda: job.epoch != self._epoch or self._stop.is_set()):
                    if job.epoch != self._epoch or self._stop.is_set():
                        cancelled = True
                        break
                    if first and job.uid not in self._first_frame:
                        first = False
                        self._first_frame.add(job.uid)
                        self.emit("tts_first_frame",
                                  uid=job.uid,
                                  ms=round((time.monotonic() - t0) * 1000, 1))
                    self.send_audio(pcm)
                    frames += 1
            except Exception:
                traceback.print_exc()
                self.emit("error", message="TTS synthesis failed")
            finally:
                if self.on_synth is not None:
                    self.on_synth(False)
            elapsed = (time.monotonic() - t0) * 1000
            with self._lock:
                cancelled = cancelled or job.epoch != self._epoch
            self._frames[job.uid] = self._frames.get(job.uid, 0) + frames
            self._synth_ms[job.uid] = self._synth_ms.get(job.uid, 0.0) + elapsed
            if job.last or cancelled:
                self._finish(job.uid, cancelled=cancelled)
        # Drain silently on shutdown.
