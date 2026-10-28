#!/usr/bin/env python3
"""Measure ZipVoice worker first-frame latency for a few target texts."""
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
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402


def read_exact(fd, n):
    buf = bytearray()
    while len(buf) < n:
        b = os.read(fd, n - len(buf))
        if not b:
            raise EOFError
        buf.extend(b)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-step", type=int, default=0)
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args()
    cfg = config.load(str(ROOT / "config/board50_zf.yaml"))
    conf = cfg["tts"]
    if args.num_step:
        conf = dict(conf)
        conf["num_step"] = args.num_step
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / ".work/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "tts_test.log", "wb", buffering=0)
    env = dict(os.environ)
    env["PATH"] = str(ROOT / "runtime/python/bin") + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT / "runtime/pylibs")
    env["LD_LIBRARY_PATH"] = str(ROOT / "runtime/pylibs") + ":/soc/lib:/opt/lib"
    r, w = os.pipe()
    args = [
        str(ROOT / conf["binary"]),
        "--model-dir", str(ROOT / conf["model_dir"]),
        "--prompt-wav", str(ROOT / conf["prompt_wav"]),
        "--prompt-text", conf["prompt_text"],
        "--vocoder-model", str(ROOT / conf["vocoder_model"]),
        "--repo-dir", str(ROOT / conf["repo_dir"]),
        "--tmp-dir", str(ROOT / conf["tmp_dir"]),
        "--response-fd", str(w),
        "--num-step", str(conf.get("num_step", 4)),
    ]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=log, stderr=log,
                            pass_fds=(w,), env=env, cwd=str(ROOT),
                            start_new_session=True)
    os.close(w)
    uid, n = struct.unpack("<II", read_exact(r, 8))
    print("ready", uid, n)

    def synth(text):
        nonlocal uid
        uid += 1
        my = uid
        t0 = time.perf_counter()
        proc.stdin.write(f"{my}\t{text}\n".encode())
        proc.stdin.flush()
        first = None
        frames = 0
        chunks = []
        while True:
            ru, rn = struct.unpack("<II", read_exact(r, 8))
            if rn == 0:
                break
            raw = read_exact(r, rn * 2)
            chunks.append(raw)
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            frames += 1
        total = (time.perf_counter() - t0) * 1000
        if save_dir is not None:
            p = save_dir / f"step{conf.get('num_step', 4)}_uid{my}.wav"
            with wave.open(str(p), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(b"".join(chunks))
            print("saved", p)
        return first, total, frames

    print("warmup", synth("Hello, this is a test."))
    for text in [
        "The process of trying wrong is simple.",
        "The blogger says that the government is not going to be able to do anything about the situation.",
        "You can hear the following special activities, he'll be big.",
        "I'm not sure what the cost of your loss is.",
    ]:
        print(text[:60], synth(text))
    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=5)
    log.close()


if __name__ == "__main__":
    main()
