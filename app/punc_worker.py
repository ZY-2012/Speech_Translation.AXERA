#!/usr/bin/python3
"""FireRedPunc resident worker (system Python 3.10 on the board).

Protocol on --response-fd:
  READY\n
  uid<TAB>punctuated-text\n
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--runtime-dir", required=True)
    ap.add_argument("--response-fd", type=int, required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.runtime_dir)
    from punc_ax import FireRedPuncAX

    p = FireRedPuncAX(args.model_dir)
    out = os.fdopen(args.response_fd, "w", encoding="utf-8", buffering=1)
    out.write("READY\n")
    out.flush()

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        if "\t" in line:
            uid, text = line.split("\t", 1)
        else:
            uid, text = "0", line
        try:
            result = p.punctuate(text, force=True)
        except Exception as exc:
            print(f"[punc-worker] {exc}", file=sys.stderr, flush=True)
            result = text
        out.write(f"{uid}\t{result}\n")
        out.flush()


if __name__ == "__main__":
    main()
