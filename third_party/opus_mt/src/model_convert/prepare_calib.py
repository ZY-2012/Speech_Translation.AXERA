#!/usr/bin/env python3
import argparse, json, os, tarfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import MarianTokenizer

ROOT = Path(os.environ.get("OPUS_MT_ROOT", Path(__file__).resolve().parents[1]))


def read_pairs(path, limit=20000):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    pairs = []
    for i in range(0, len(lines) - 3, 4):
        src, ref = lines[i], lines[i + 1]
        if src and ref:
            pairs.append((src, ref))
        if len(pairs) >= limit:
            break
    return pairs


def make_encoder_inputs(tok, text, src_len, pad_id):
    ids = tok([text], return_tensors="np")["input_ids"].astype(np.int32)[:, :src_len]
    mask = np.ones_like(ids, dtype=np.float32)
    if ids.shape[1] < src_len:
        p = src_len - ids.shape[1]
        ids = np.pad(ids, ((0, 0), (0, p)), constant_values=pad_id)
        mask = np.pad(mask, ((0, 0), (0, p)), constant_values=0)
    return ids, mask


def target_ids(tok, text, max_len):
    pieces = tok.spm_target.encode(text, out_type=str)
    ids = tok.convert_tokens_to_ids(pieces)
    return [int(x) for x in ids][: max_len - 1]


def save_tar(out_dir, name, arrays):
    tmp = out_dir / name
    tmp.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(arrays):
        np.save(tmp / f"{i:05d}.npy", a)
    tar_path = out_dir / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(tmp.glob("*.npy")):
            tar.add(f, arcname=f.name)
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=["zh-en", "en-zh"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--src-len", type=int, default=128)
    ap.add_argument("--tgt-len", type=int, default=64)
    args = ap.parse_args()
    d = args.direction
    export_dir = ROOT / d / "export"
    tok = MarianTokenizer.from_pretrained(str(export_dir))
    meta = json.loads((export_dir / "model_meta.json").read_text())
    pad_id = meta["pad_token_id"]
    start_id = meta["decoder_start_token_id"]
    src_len, tgt_len = args.src_len, args.tgt_len

    pairs = read_pairs(Path(__file__).resolve().parent / "data" / f"{d}.test.txt")
    step = max(1, len(pairs) // (args.n * 4))
    pairs = pairs[::step]

    enc_sess = ort.InferenceSession(str(export_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
    out_dir = export_dir / "calib_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_ids_list, enc_mask_list, hid_list = [], [], []
    dec_ids_list, dec_attn_list, pos_list, dec_encmask_list = [], [], [], []
    sources, refs = [], []
    used = 0
    for src, ref in pairs:
        ei, em = make_encoder_inputs(tok, src, src_len, pad_id)
        if em.sum() < 2:
            continue
        hid = enc_sess.run(None, {"input_ids": ei, "attention_mask": em})[0]
        tids = target_ids(tok, ref, tgt_len)
        if not tids:
            continue
        L = min(len(tids) + 1, tgt_len)
        # two prefixes: early step and a random later step
        lens = sorted({1, L, max(1, L // 2), max(1, int(L * 0.75))})
        for t in lens:
            dec = np.full((1, tgt_len), pad_id, dtype=np.int32)
            dec[0, tgt_len - t:] = [start_id] + tids[: t - 1]
            dm = np.zeros((1, tgt_len), dtype=np.int32)
            dm[0, tgt_len - t:] = 1
            pos = np.zeros((1, tgt_len), dtype=np.int32)
            pos[0, tgt_len - t:] = np.arange(t, dtype=np.int32)
            dec_ids_list.append(dec)
            dec_attn_list.append(dm)
            pos_list.append(pos)
            dec_encmask_list.append(em.copy())
        enc_ids_list.append(ei)
        enc_mask_list.append(em)
        hid_list.append(hid)
        sources.append(src)
        refs.append(ref)
        used += 1
        if used >= args.n:
            break

    print(f"[{d}] calib samples: enc={len(enc_ids_list)} dec={len(dec_ids_list)}")
    save_tar(out_dir, "input_ids", enc_ids_list)
    save_tar(out_dir, "attention_mask", enc_mask_list)
    save_tar(out_dir, "encoder_hidden_states", hid_list)
    save_tar(out_dir, "decoder_input_ids", dec_ids_list)
    save_tar(out_dir, "decoder_attention_mask", dec_attn_list)
    save_tar(out_dir, "position_ids", pos_list)
    save_tar(out_dir, "decoder_encoder_attention_mask", dec_encmask_list)
    (out_dir / "manifest.json").write_text(json.dumps({
        "direction": d, "n_enc": len(enc_ids_list), "n_dec": len(dec_ids_list),
        "sources": sources, "references": refs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in sorted(out_dir.glob("*.tar.gz")):
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
