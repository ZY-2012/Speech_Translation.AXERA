#!/usr/bin/env python3
"""Direct test of third_party/zf/zf_asr_worker.py on a WAV file."""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_exact(fd, n):
    buf = bytearray()
    while len(buf) < n:
        b = os.read(fd, n - len(buf))
        if not b:
            raise EOFError
        buf.extend(b)
    return bytes(buf)


def read_frame(fd):
    typ = read_exact(fd, 1)
    (n,) = struct.unpack("<I", read_exact(fd, 4))
    return typ, read_exact(fd, n) if n else b""


def drain(fd, events):
    os.set_blocking(fd, False)
    try:
        while True:
            try:
                typ, payload = read_frame(fd)
            except (BlockingIOError, EOFError):
                break
            if typ == b"T" and len(payload) >= 8:
                flags, _ = struct.unpack("<II", payload[:8])
                events.append((flags, payload[8:].decode("utf-8", "replace")))
    finally:
        os.set_blocking(fd, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(ROOT / "samples/zh_6s.wav"))
    ap.add_argument("--model-dir", default=str(ROOT / "../../ZF/streaming/release/models"))
    ap.add_argument("--chunk-bytes", type=int, default=640)
    args = ap.parse_args()

    log_dir = ROOT / ".work/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "zf_test.log", "wb", buffering=0)
    r, w = os.pipe()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/soc/lib:/opt/lib:" + env.get("LD_LIBRARY_PATH", "")
    args_list = [
        str(ROOT / "third_party/zf/zf_asr_worker.py"),
        "--model-dir", args.model_dir,
        "--language", "zh",
        "--response-fd", str(w),
    ]
    t0 = time.time()
    proc = subprocess.Popen(args_list, stdin=subprocess.PIPE, stdout=log,
                            stderr=log, pass_fds=(w,), env=env,
                            start_new_session=True)
    os.close(w)
    typ, payload = read_frame(r)
    print(f"ready={typ} in {time.time()-t0:.2f}s")
    if typ != b"R":
        print("bad ready", typ, payload)
        return

    with wave.open(args.wav, "rb") as wf:
        rate = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    print(f"wav={args.wav} rate={rate} ch={ch} bytes={len(raw)}")
    events = []
    for i in range(0, len(raw), args.chunk_bytes):
        chunk = raw[i:i + args.chunk_bytes]
        proc.stdin.write(struct.pack("<I", len(chunk)))
        proc.stdin.write(chunk)
        proc.stdin.flush()
        drain(r, events)
        time.sleep(0.02)
    t_flush = time.time()
    proc.stdin.write(struct.pack("<I", 0))
    proc.stdin.flush()
    typ, payload = read_frame(r)
    if typ == b"T" and len(payload) >= 8:
        flags, _ = struct.unpack("<II", payload[:8])
        text = payload[8:].decode("utf-8", "replace")
        print(f"final flags={flags} text={text!r} flush={time.time()-t_flush:.2f}s")
    else:
        print("bad final", typ, payload)
    print(f"partials={len(events)} last={events[-1][1] if events else None!r}")
    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=5)
    log.close()


if __name__ == "__main__":
    main()
