#!/usr/bin/env python3
"""Generate cat_tokens binary for C++ ZipVoice inference. Works for any language.

Usage:
  python3 gen_cat_tokens.py --prompt "prompt text" --text "target text"
  python3 gen_cat_tokens.py --prompt-file prompt.txt --text-file target.txt
"""
import sys, os, argparse
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cpp/scripts → repo root
sys.path.insert(0, REPO_DIR)

from scripts.local_tokenizer import LocalEmiliaTokenizer
from scripts.text_processing import normalize_punctuation

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', default='')
parser.add_argument('--text', default='')
parser.add_argument('--prompt-file', default='')
parser.add_argument('--text-file', default='')
parser.add_argument('--output', default='cat_tokens.bin')
parser.add_argument('--max-tokens', type=int, default=384)
args = parser.parse_args()

prompt = args.prompt or (open(args.prompt_file).read().strip() if args.prompt_file else '')
text = args.text or (open(args.text_file).read().strip() if args.text_file else '')

if not prompt or not text:
    print("ERROR: provide --prompt/--prompt-file and --text/--text-file")
    sys.exit(1)

token_file = os.path.join(REPO_DIR, 'resources', 'zipvoice_hf', 'zipvoice', 'tokens.txt')
tokenizer = LocalEmiliaTokenizer(token_file=token_file)

pids = tokenizer.texts_to_token_ids([normalize_punctuation(prompt)])[0]
tids = tokenizer.texts_to_token_ids([normalize_punctuation(text)])[0]

cat = pids + tids + [tokenizer.pad_id]
ct = np.full((args.max_tokens,), tokenizer.pad_id, dtype=np.int32)
ct[:len(cat)] = np.array(cat, dtype=np.int32)
ct.tofile(args.output)

print(f'prompt_tokens_len={len(pids)}  text_tokens_len={len(tids)}  saved: {args.output}')
