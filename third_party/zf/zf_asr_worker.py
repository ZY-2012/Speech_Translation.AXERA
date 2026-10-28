#!/usr/bin/python3
"""Streaming Zipformer ASR worker for the Speech_Translation pipeline.

Protocol matches third_party/asr/bin/stream-asr-worker:
  stdin: uint32 n + n bytes S16LE PCM; n == 0 flushes the utterance
  fd:    'R' + uint32 0
         'T' + uint32 len + uint32 flags + uint32 coverage + UTF-8 text
         'E' + uint32 len + UTF-8 error

The encoder/decoder/joiner are the AX650 U16 release models from
ASR/ZF/streaming/release/models.  The worker is run with the system Python
3.10 because it provides kaldi_native_fbank and axengine.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import traceback
from pathlib import Path

import numpy as np


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed")
        buf.extend(chunk)
    return bytes(buf)


def load_tokens(path: Path) -> dict[int, str]:
    table: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            token, _, idx = line.rstrip("\n").rpartition(" ")
            if token:
                table[int(idx)] = token
    return table


def tokens_to_text(tokens: list[int], table: dict[int, str]) -> str:
    return "".join(table.get(i, "") for i in tokens).replace("\u2581", " ").strip()


class StreamingZipformer:
    def __init__(self, model_dir: Path, chunk: int = 32):
        from axengine import InferenceSession

        self.chunk = int(chunk)
        self.T = self.chunk + 7
        self.blank = 0
        self.context = 2
        self.enc = InferenceSession(
            str(model_dir / "encoder.axmodel"),
            providers=["AxEngineExecutionProvider"])
        self.dec = InferenceSession(
            str(model_dir / "decoder.axmodel"),
            providers=["AxEngineExecutionProvider"])
        self.join = InferenceSession(
            str(model_dir / "joiner.axmodel"),
            providers=["AxEngineExecutionProvider"])
        self.tokens = load_tokens(model_dir / "tokens.txt")

        self.enc_out_names = [o.name for o in self.enc.get_outputs()]
        self.state_names = [i.name for i in self.enc.get_inputs()
                            if i.name != "x"]
        self.state_shapes = {i.name: list(i.shape)
                             for i in self.enc.get_inputs()}
        self.dec_in_name = self.dec.get_inputs()[0].name
        self.dec_out_name = self.dec.get_outputs()[0].name
        self.join_in_names = [i.name for i in self.join.get_inputs()]
        self.join_out_name = self.join.get_outputs()[0].name
        self.reset()

    def reset(self):
        self.states = {}
        for name in self.state_names:
            shape = self.state_shapes[name]
            dtype = np.int32 if name.startswith("cached_len") else np.float32
            self.states[name] = np.zeros(shape, dtype=dtype)
        self.hyp: list[int] = [self.blank] * self.context
        self.dec_out = self._run_decoder()
        self.feat_buf = np.zeros((0, 80), dtype=np.float32)
        self.fbank = self._make_fbank()
        self.fbank_read = 0
        self.last_text = ""

    @staticmethod
    def _make_fbank():
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.window_type = "povey"
        opts.mel_opts.num_bins = 80
        opts.mel_opts.low_freq = 20.0
        opts.mel_opts.high_freq = -400.0
        opts.energy_floor = 0.0
        return knf.OnlineFbank(opts)

    def _run_decoder(self) -> np.ndarray:
        y = np.asarray([self.hyp[-self.context:]], dtype=np.int32)
        return self.dec.run([self.dec_out_name], {self.dec_in_name: y})[0]

    def _run_chunk(self, seg: np.ndarray):
        feed = {"x": np.ascontiguousarray(seg[None, :, :], dtype=np.float32)}
        feed.update(self.states)
        outs = self.enc.run(self.enc_out_names, feed)
        out = dict(zip(self.enc_out_names, outs))
        for name in self.state_names:
            self.states[name] = out["new_" + name]
        enc_out = out["encoder_out"]
        for t in range(enc_out.shape[1]):
            cur = enc_out[:, t:t + 1, :].reshape(1, -1).astype(np.float32)
            logits = self.join.run(
                [self.join_out_name],
                {self.join_in_names[0]: cur,
                 self.join_in_names[1]: self.dec_out},
            )[0]
            sym = int(logits.reshape(-1).argmax())
            if sym != self.blank:
                self.hyp.append(sym)
                self.dec_out = self._run_decoder()

    def _consume_frames(self):
        n = int(self.fbank.num_frames_ready)
        if n <= self.fbank_read:
            return
        frames = [self.fbank.get_frame(i)
                  for i in range(self.fbank_read, n)]
        self.fbank_read = n
        arr = np.asarray(frames, dtype=np.float32)
        if self.feat_buf.size:
            self.feat_buf = np.concatenate([self.feat_buf, arr], axis=0)
        else:
            self.feat_buf = arr
        while self.feat_buf.shape[0] >= self.T:
            seg = self.feat_buf[:self.T]
            self._run_chunk(seg)
            self.feat_buf = self.feat_buf[self.chunk:]

    def feed_pcm(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if audio.size:
            self.fbank.accept_waveform(16000, audio)
            self._consume_frames()
        return self.text()

    def flush(self) -> str:
        self.fbank.input_finished()
        self._consume_frames()
        if self.feat_buf.shape[0]:
            pad = np.zeros((self.T - self.feat_buf.shape[0], 80),
                           dtype=np.float32)
            seg = np.concatenate([self.feat_buf, pad], axis=0)
            self._run_chunk(seg)
        text = self.text()
        self.reset()
        return text

    def text(self) -> str:
        return tokens_to_text(self.hyp[self.context:], self.tokens)


def write_frame(fd: int, typ: bytes, payload: bytes = b""):
    os.write(fd, typ + struct.pack("<I", len(payload)) + payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--language", default="auto")
    ap.add_argument("--response-fd", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    engine = StreamingZipformer(model_dir, chunk=args.chunk)
    fd = args.response_fd
    write_frame(fd, b"R", b"")

    stdin = sys.stdin.buffer
    try:
        while True:
            (n,) = struct.unpack("<I", read_exact(stdin, 4))
            if n == 0:
                text = engine.flush()
                payload = struct.pack("<II", 2, 0) + text.encode("utf-8")
                write_frame(fd, b"T", payload)
                continue
            pcm = read_exact(stdin, n)
            text = engine.feed_pcm(pcm)
            if text != engine.last_text:
                engine.last_text = text
                payload = struct.pack("<II", 1, 0) + text.encode("utf-8")
                write_frame(fd, b"T", payload)
    except EOFError:
        pass
    except Exception:
        traceback.print_exc()
        try:
            write_frame(fd, b"E", b"zipformer worker failed")
        except Exception:
            pass


if __name__ == "__main__":
    main()
