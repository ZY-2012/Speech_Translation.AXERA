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

# The C++ worker rewrites the prompt file on every request, but the content is
# fixed.  Cache the prompt token ids so only the new target text is tokenized.
_prompt_cache = {"text": None, "ids": None}


def get_prompt_ids(prompt_file):
    global _prompt_cache
    pt = normalize_punctuation(open(prompt_file).read().strip())
    if pt != _prompt_cache["text"]:
        _prompt_cache["text"] = pt
        _prompt_cache["ids"] = tokenizer.texts_to_token_ids([pt])[0]
    return _prompt_cache["ids"]


for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("\t")
    cmd = parts[0]

    if cmd == "count":
        prompt_file, text_file = parts[1], parts[2]
        pids = get_prompt_ids(prompt_file)
        tt = normalize_punctuation(open(text_file).read().strip())
        tids = tokenizer.texts_to_token_ids([tt])[0]
        print(f"COUNT {len(pids)} {len(tids)}", flush=True)

    elif cmd == "tokenize":
        prompt_file, text_file, max_tokens, output_bin = parts[1], parts[2], int(parts[3]), parts[4]
        pids = get_prompt_ids(prompt_file)
        tt = normalize_punctuation(open(text_file).read().strip())
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
