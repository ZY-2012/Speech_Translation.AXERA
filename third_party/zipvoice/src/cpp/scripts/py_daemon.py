#!/usr/bin/env python3
"""
ZipVoice Python daemon: loads tokenizer once, handles C++ requests via stdin/stdout.
Protocol (tab-separated lines to preserve spaces in text paths):
  count\t<prompt_file>\t<text_file>
    -> prints "COUNT <prompt_len> <text_len>"
  tokenize\t<prompt_file>\t<text_file>\t<max_tokens>\t<output_bin>
    -> prints "TOKENS <prompt_len> <text_len>"
  quit
    -> exits
"""
import sys, os
import numpy as np

REPO_DIR = sys.argv[1]
sys.path.insert(0, REPO_DIR)

from scripts.local_tokenizer import LocalEmiliaTokenizer
from scripts.text_processing import normalize_punctuation

TOKEN_FILE = os.path.join(REPO_DIR, "resources", "zipvoice_hf", "zipvoice", "tokens.txt")
tokenizer = LocalEmiliaTokenizer(token_file=TOKEN_FILE)

print("READY", flush=True)

for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("\t")
    cmd = parts[0]

    if cmd == "count":
        prompt_file, text_file = parts[1], parts[2]
        pt = normalize_punctuation(open(prompt_file).read().strip())
        tt = normalize_punctuation(open(text_file).read().strip())
        pids = tokenizer.texts_to_token_ids([pt])[0]
        tids = tokenizer.texts_to_token_ids([tt])[0]
        print(f"COUNT {len(pids)} {len(tids)}", flush=True)

    elif cmd == "tokenize":
        prompt_file, text_file, max_tokens, output_bin = parts[1], parts[2], int(parts[3]), parts[4]
        pt = normalize_punctuation(open(prompt_file).read().strip())
        tt = normalize_punctuation(open(text_file).read().strip())
        pids = tokenizer.texts_to_token_ids([pt])[0]
        tids = tokenizer.texts_to_token_ids([tt])[0]
        cat = pids + tids + [tokenizer.pad_id]
        if len(cat) > max_tokens:
            print(f"ERROR too_many_tokens {len(cat)}>{max_tokens}", flush=True)
            continue
        ct = np.full((max_tokens,), tokenizer.pad_id, dtype=np.int32)
        ct[:len(cat)] = np.array(cat, dtype=np.int32)
        ct.tofile(output_bin)
        print(f"TOKENS {len(pids)} {len(tids)}", flush=True)

    elif cmd == "quit":
        break
