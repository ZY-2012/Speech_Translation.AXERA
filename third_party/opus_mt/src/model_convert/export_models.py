#!/usr/bin/env python3
import argparse, json, os, shutil, sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

ROOT = Path(os.environ.get("OPUS_MT_ROOT", Path(__file__).resolve().parents[1]))


class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state


class DecoderWrapper(nn.Module):
    """Decoder step with left-padded prefix and explicit position ids.

    The last slot always holds the current token, so the LM head runs on one
    position only (no dynamic Gather, much smaller output and compute).
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.register_buffer("final_logits_bias", model.final_logits_bias.clone().view(-1))

    def forward(self, decoder_input_ids, decoder_attention_mask, position_ids,
                encoder_hidden_states, encoder_attention_mask):
        dec = self.decoder
        hidden = dec.embed_tokens(decoder_input_ids) * dec.embed_scale
        bsz, tgt_len = decoder_input_ids.shape
        hidden = hidden + dec.embed_positions((bsz, tgt_len), 0, position_ids=position_ids)

        neg = -1e4
        causal = torch.triu(torch.full((tgt_len, tgt_len), neg, dtype=hidden.dtype, device=hidden.device), diagonal=1)
        pad = (1.0 - decoder_attention_mask.to(hidden.dtype)) * neg
        self_mask = causal.unsqueeze(0).unsqueeze(0) + pad[:, None, None, :]
        enc_mask = (1.0 - encoder_attention_mask.to(hidden.dtype))[:, None, None, :] * neg

        for layer in dec.layers:
            hidden = layer(hidden, self_mask, encoder_hidden_states,
                           encoder_attention_mask=enc_mask, past_key_values=None, use_cache=False)
        hidden = hidden[:, -1:, :]
        return self.lm_head(hidden) + self.final_logits_bias


def load_model(model_dir, device="cpu"):
    tok = MarianTokenizer.from_pretrained(str(model_dir))
    model = MarianMTModel.from_pretrained(str(model_dir), attn_implementation="eager").eval().to(device)
    return tok, model


def export_one(module, args, path, input_names, output_names, dynamic_axes=None):
    with torch.no_grad():
        torch.onnx.export(
            module, args, str(path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=14,
            do_constant_folding=True, dynamo=False,
        )


def simplify(raw, out_dir):
    import onnxsim, onnxslim
    sim = out_dir / (raw.stem + ".sim.onnx")
    slim = out_dir / (raw.stem.replace(".raw", "") + ".onnx")
    m = onnx.load(str(raw))
    m, ok = onnxsim.simplify(m)
    assert ok
    onnx.save(m, str(sim))
    m2 = onnxslim.slim(str(sim))
    onnx.save(m2, str(slim))
    onnx.checker.check_model(str(slim))
    return raw, sim, slim


def make_encoder_inputs(tok, text, src_len, pad_id):
    ids = tok([text], return_tensors="np")["input_ids"].astype(np.int32)
    ids = ids[:, :src_len]
    mask = np.ones_like(ids, dtype=np.float32)
    if ids.shape[1] < src_len:
        pad = src_len - ids.shape[1]
        ids = np.pad(ids, ((0, 0), (0, pad)), constant_values=pad_id)
        mask = np.pad(mask, ((0, 0), (0, pad)), constant_values=0)
    return ids, mask


def onnx_greedy(enc_sess, dec_sess, enc_in, start_id, eos_id, bad_ids, max_new, pad_id):
    enc_hidden = enc_sess.run(None, {"input_ids": enc_in[0], "attention_mask": enc_in[1]})[0]
    ids = [start_id]
    for step in range(max_new):
        t = len(ids)
        dec_ids = np.full((1, max_new), pad_id, dtype=np.int32)
        dec_ids[0, max_new - t:] = ids
        dec_mask = np.zeros((1, max_new), dtype=np.int32)
        dec_mask[0, max_new - t:] = 1
        pos = np.zeros((1, max_new), dtype=np.int32)
        pos[0, max_new - t:] = np.arange(t, dtype=np.int32)
        logits = dec_sess.run(None, {
            "decoder_input_ids": dec_ids,
            "decoder_attention_mask": dec_mask,
            "position_ids": pos,
            "encoder_hidden_states": enc_hidden,
            "encoder_attention_mask": enc_in[1],
        })[0][0, 0]
        nxt = int(np.argmax(logits))
        if nxt in bad_ids:
            logits[nxt] = -np.inf
            nxt = int(np.argmax(logits))
        ids.append(nxt)
        if nxt == eos_id:
            break
    return ids


def torch_greedy(model, tok, text, max_new):
    b = tok([text], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(**b, max_new_tokens=max_new, num_beams=1, do_sample=False)
    return out[0].tolist()


def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=["zh-en", "en-zh"])
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--src-len", type=int, default=128)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--no-slim", action="store_true")
    args = ap.parse_args()

    direction = args.direction
    model_dir = Path(args.model_dir or ROOT / "origin" / direction)
    out_dir = Path(args.out or ROOT / direction / "export")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(model_dir)
    cfg = model.config
    pad_id = cfg.pad_token_id
    eos_id = cfg.eos_token_id
    start_id = cfg.decoder_start_token_id
    bad_ids = {pad_id}

    print(f"[{direction}] vocab={cfg.vocab_size} d={cfg.d_model} layers={cfg.encoder_layers} pad={pad_id} eos={eos_id} start={start_id}")

    enc = EncoderWrapper(model).eval()
    dec = DecoderWrapper(model).eval()

    sample_text = "这是一个测试。" if direction == "zh-en" else ">>cmn_Hans<< This is a test."
    enc_ids, enc_mask = make_encoder_inputs(tok, sample_text, args.src_len, pad_id)
    enc_ids_t = torch.from_numpy(enc_ids)
    enc_mask_t = torch.from_numpy(enc_mask)
    with torch.no_grad():
        ref_enc = enc(enc_ids_t, enc_mask_t).numpy()

    dec_prefix = np.full((1, args.tgt_len), pad_id, dtype=np.int32)
    dec_prefix[0, -1] = start_id
    dec_mask = np.zeros((1, args.tgt_len), dtype=np.int32)
    dec_mask[0, -1] = 1
    pos_ids = np.zeros((1, args.tgt_len), dtype=np.int32)
    dec_prefix_t = torch.from_numpy(dec_prefix)
    dec_mask_t = torch.from_numpy(dec_mask)
    pos_t = torch.from_numpy(pos_ids)
    enc_hidden_t = torch.from_numpy(ref_enc)
    with torch.no_grad():
        ref_dec = dec(dec_prefix_t, dec_mask_t, pos_t, enc_hidden_t, enc_mask_t).numpy()

    raw_enc = out_dir / "encoder.raw.onnx"
    raw_dec = out_dir / "decoder.raw.onnx"
    export_one(enc, (enc_ids_t, enc_mask_t), raw_enc,
               ["input_ids", "attention_mask"], ["encoder_hidden_states"])
    export_one(dec, (dec_prefix_t, dec_mask_t, pos_t, enc_hidden_t, enc_mask_t), raw_dec,
               ["decoder_input_ids", "decoder_attention_mask", "position_ids",
                "encoder_hidden_states", "encoder_attention_mask"],
               ["logits"])

    if args.no_slim:
        shutil.copy(raw_enc, out_dir / "encoder.onnx")
        shutil.copy(raw_dec, out_dir / "decoder.onnx")
        enc_onnx, dec_onnx = out_dir / "encoder.onnx", out_dir / "decoder.onnx"
    else:
        e_raw, e_sim, e_slim = simplify(raw_enc, out_dir)
        d_raw, d_sim, d_slim = simplify(raw_dec, out_dir)
        enc_onnx, dec_onnx = e_slim, d_slim
        for p in [e_raw, e_sim, d_raw, d_sim]:
            print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")
        print(f"  encoder.onnx: {enc_onnx.stat().st_size/1e6:.1f} MB")
        print(f"  decoder.onnx: {dec_onnx.stat().st_size/1e6:.1f} MB")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc_sess = ort.InferenceSession(str(enc_onnx), so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_onnx), so, providers=["CPUExecutionProvider"])

    got_enc = enc_sess.run(None, {"input_ids": enc_ids, "attention_mask": enc_mask})[0]
    got_dec = dec_sess.run(None, {
        "decoder_input_ids": dec_prefix,
        "decoder_attention_mask": dec_mask,
        "position_ids": pos_ids,
        "encoder_hidden_states": ref_enc,
        "encoder_attention_mask": enc_mask,
    })[0]
    print(f"  encoder cos={cosine(ref_enc, got_enc):.6f}  decoder cos={cosine(ref_dec, got_dec):.6f}")

    samples = [
        ("这是一个测试。", "It's a test."),
        ("今天天气很好。", "The weather is nice today."),
        ("我正在学习人工智能。", "I am studying artificial intelligence."),
        ("他昨天去了北京。", "He went to Beijing yesterday."),
    ] if direction == "zh-en" else [
        (">>cmn_Hans<< This is a test.", "这是一个测试。"),
        (">>cmn_Hans<< The weather is nice today.", "今天天气很好。"),
        (">>cmn_Hans<< I am studying artificial intelligence.", "我正在学习人工智能。"),
        (">>cmn_Hans<< He went to Beijing yesterday.", "他昨天去了北京。"),
    ]
    exact = 0
    for src, _ in samples:
        ei, em = make_encoder_inputs(tok, src, args.src_len, pad_id)
        o_ids = onnx_greedy(enc_sess, dec_sess, (ei, em), start_id, eos_id, bad_ids, args.tgt_len, pad_id)
        t_ids = torch_greedy(model, tok, src, args.tgt_len)
        o_txt = tok.decode(o_ids, skip_special_tokens=True)
        t_txt = tok.decode(t_ids, skip_special_tokens=True)
        same = o_txt.strip() == t_txt.strip()
        exact += same
        print(f"  {'OK ' if same else 'DIFF'} onnx={o_txt!r} torch={t_txt!r}")
    print(f"  exact match {exact}/{len(samples)}")

    meta = {
        "direction": direction,
        "model_dir": str(model_dir),
        "src_len": args.src_len,
        "tgt_len": args.tgt_len,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "decoder_start_token_id": start_id,
        "vocab_size": cfg.vocab_size,
        "encoder": {
            "file": enc_onnx.name,
            "inputs": [
                {"name": "input_ids", "shape": [1, args.src_len], "dtype": "int32"},
                {"name": "attention_mask", "shape": [1, args.src_len], "dtype": "float32"},
            ],
            "outputs": [{"name": "encoder_hidden_states", "shape": [1, args.src_len, cfg.d_model], "dtype": "float32"}],
        },
        "decoder": {
            "file": dec_onnx.name,
            "inputs": [
                {"name": "decoder_input_ids", "shape": [1, args.tgt_len], "dtype": "int32"},
                {"name": "decoder_attention_mask", "shape": [1, args.tgt_len], "dtype": "int32"},
                {"name": "position_ids", "shape": [1, args.tgt_len], "dtype": "int32"},
                {"name": "encoder_hidden_states", "shape": [1, args.src_len, cfg.d_model], "dtype": "float32"},
                {"name": "encoder_attention_mask", "shape": [1, args.src_len], "dtype": "float32"},
            ],
            "outputs": [{"name": "logits", "shape": [1, 1, cfg.vocab_size], "dtype": "float32"}],
            "decode_layout": "left_padded_prefix, last slot = current token",
        },
        "enc_cos": cosine(ref_enc, got_enc),
        "dec_cos": cosine(ref_dec, got_dec),
        "exact_match": f"{exact}/{len(samples)}",
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for f in ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json", "config.json", "generation_config.json"]:
        shutil.copy(model_dir / f, out_dir / f)
    print(f"[{direction}] export done -> {out_dir}")


if __name__ == "__main__":
    main()
