#!/usr/bin/env python3
"""OPUS-MT (Marian) translation on AX650, sentencepiece-only tokenizer."""
import json
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

DEFAULT_PREFIX = ">>cmn_Hans<< "


class Tokenizer:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.piece_to_id = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self.id_to_piece = {v: k for k, v in self.piece_to_id.items()}
        self.spm_src = spm.SentencePieceProcessor(model_file=str(d / "source.spm"))
        self.spm_tgt = spm.SentencePieceProcessor(model_file=str(d / "target.spm"))
        cfg = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        self.eos = int(cfg["eos_token_id"])
        self.pad = int(cfg["pad_token_id"])
        self.start = int(cfg["decoder_start_token_id"])
        self.src_len = int(cfg["src_len"])
        self.tgt_len = int(cfg["tgt_len"])
        self.unk = self.piece_to_id.get("<unk>", 1)
        self.special = {self.eos, self.pad, self.unk}

    def encode(self, text):
        code = None
        rest = text
        if text.startswith(">>"):
            end = text.find("<<")
            if end != -1:
                code = text[: end + 2]
                rest = text[end + 2 :]
        pieces = ([code] if code else []) + self.spm_src.encode(rest, out_type=str)
        ids = [self.piece_to_id.get(p, self.unk) for p in pieces]
        truncated = len(ids) > self.src_len - 1
        ids = ids[: self.src_len - 1] + [self.eos]
        return ids, truncated

    def decode(self, ids):
        pieces = []
        for i in ids:
            i = int(i)
            if i in self.special:
                continue
            p = self.id_to_piece.get(i)
            if p is None or (p.startswith(">>") and p.endswith("<<")):
                continue
            pieces.append(p)
        return self.spm_src.decode_pieces(pieces).replace("\u2581", " ").strip()


class Translator:
    def __init__(self, direction, model_root=None, max_new_tokens=None):
        if direction not in ("zh-en", "en-zh"):
            raise ValueError("direction must be zh-en or en-zh")
        self.direction = direction
        root = Path(model_root) if model_root else Path(__file__).resolve().parents[1] / "models"
        self.model_dir = Path(root) / direction
        self.tok = Tokenizer(self.model_dir)
        import axengine as axe
        self.enc = axe.InferenceSession(str(self.model_dir / "encoder.axmodel"))
        self.dec = axe.InferenceSession(str(self.model_dir / "decoder.axmodel"))
        self.max_new_tokens = int(max_new_tokens or self.tok.tgt_len)
        self.warmup()

    def _prepare(self, text):
        if self.direction == "en-zh" and not text.startswith(">>"):
            text = DEFAULT_PREFIX + text
        ids, truncated = self.tok.encode(text)
        n = len(ids)
        x = np.full((1, self.tok.src_len), self.tok.pad, dtype=np.int32)
        x[0, :n] = ids
        m = np.zeros((1, self.tok.src_len), dtype=np.float32)
        m[0, :n] = 1.0
        return x, m, truncated

    def warmup(self):
        for _ in range(2):
            self.translate("你好" if self.direction == "zh-en" else "Hello")

    def translate(self, text, return_stats=False):
        if not text.strip():
            return ("", {"src_tokens": 0, "out_tokens": 0, "enc_ms": 0.0,
                         "dec_ms": 0.0, "dec_step_ms": 0.0, "truncated": False}) if return_stats else ""
        x, m, truncated = self._prepare(text)
        t0 = time.perf_counter()
        hidden = self.enc.run(None, {"input_ids": x, "attention_mask": m})[0]
        t_enc = time.perf_counter() - t0

        out = [self.tok.start]
        dec_times = []
        for _ in range(self.max_new_tokens):
            t = len(out)
            dec = np.full((1, self.tok.tgt_len), self.tok.pad, dtype=np.int32)
            dec[0, self.tok.tgt_len - t :] = out
            dm = np.zeros((1, self.tok.tgt_len), dtype=np.int32)
            dm[0, self.tok.tgt_len - t :] = 1
            pos = np.zeros((1, self.tok.tgt_len), dtype=np.int32)
            pos[0, self.tok.tgt_len - t :] = np.arange(t, dtype=np.int32)
            t0 = time.perf_counter()
            logits = self.dec.run(None, {
                "decoder_input_ids": dec,
                "decoder_attention_mask": dm,
                "position_ids": pos,
                "encoder_hidden_states": hidden,
                "encoder_attention_mask": m,
            })[0][0, 0].astype(np.float32)
            dec_times.append(time.perf_counter() - t0)
            logits[self.tok.pad] = -np.inf
            nxt = int(np.argmax(logits))
            out.append(nxt)
            if nxt == self.tok.eos:
                break
        result = self.tok.decode(out[1:])
        if not return_stats:
            return result
        return result, {
            "src_tokens": int(m.sum()),
            "out_tokens": len(out) - 1,
            "enc_ms": t_enc * 1000,
            "dec_ms": float(np.sum(dec_times)) * 1000,
            "dec_step_ms": float(np.mean(dec_times)) * 1000 if dec_times else 0.0,
            "truncated": truncated,
        }
