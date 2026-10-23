#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from opus_mt_ax import Translator


def main():
    ap = argparse.ArgumentParser(description="OPUS-MT zh<->en translation on AX650")
    ap.add_argument("--direction", required=True, choices=["zh-en", "en-zh"])
    ap.add_argument("--text")
    ap.add_argument("--input", help="UTF-8 text file, one sentence per line")
    ap.add_argument("--output", help="output file for --input mode")
    ap.add_argument("--model-root")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    tr = Translator(args.direction, args.model_root, args.max_new_tokens)
    if args.text is not None:
        out, st = tr.translate(args.text, True)
        print(out)
        if args.stats:
            print(st)
        return

    srcs = [l for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    outs, stats = [], []
    t0 = time.perf_counter()
    for s in srcs:
        o, st = tr.translate(s, True)
        outs.append(o)
        stats.append(st)
    dt = time.perf_counter() - t0
    if args.output:
        Path(args.output).write_text("\n".join(outs) + "\n", encoding="utf-8")
    n = max(len(outs), 1)
    print(f"translated {len(outs)} lines in {dt:.2f}s ({dt/n*1000:.1f} ms/line)")
    if args.stats and stats:
        print(f"avg encoder {np.mean([s['enc_ms'] for s in stats]):.2f} ms | "
              f"decoder {np.mean([s['dec_ms'] for s in stats]):.2f} ms | "
              f"step {np.mean([s['dec_step_ms'] for s in stats]):.2f} ms | "
              f"tokens/line {np.mean([s['out_tokens'] for s in stats]):.1f}")


if __name__ == "__main__":
    main()
