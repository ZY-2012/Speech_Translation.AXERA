#!/usr/bin/env python3
"""
ZipVoice static ONNX inference verification.

Pipeline:
  encoder_core.onnx (static) → host duration expansion →
  fm_decoder.onnx (static, ODE 16-step) → vocoder → waveform

Usage:
    python infer_onnx_static.py \
        --model-dir ./models/static_onnx \
        --prompt-wav prompt.wav \
        --prompt-text "I am a prompt." \
        --text "I am a sentence to synthesize." \
        --output-wav result_static.wav
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

# Add local runtime to path. ZipVoice itself is imported from the Python environment.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for p in [str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from onnx_zipvoice_runtime import StaticOnnxZipVoiceRuntime
from zipvoice.bin.infer_zipvoice import get_vocoder
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.common import str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    load_prompt_wav,
    remove_silence,
    rms_norm,
)


def setup_jieba_cache() -> None:
    import jieba

    cache_dir = _HERE.parent / ".work_tmp" / "jieba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(cache_dir)
    jieba.dt.cache_file = "jieba.cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZipVoice static ONNX inference verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing static ONNX models and runtime_config.json")
    parser.add_argument("--prompt-wav", required=True,
                        help="Path to prompt/reference audio")
    parser.add_argument("--prompt-text", required=True,
                        help="Transcription of prompt audio")
    parser.add_argument("--text", required=True,
                        help="Text to synthesize")
    parser.add_argument("--output-wav", default="output_static.wav",
                        help="Output WAV path")
    parser.add_argument("--num-step", type=int, default=16,
                        help="Number of ODE sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed control")
    parser.add_argument("--t-shift", type=float, default=0.5,
                        help="Time shift for ODE solver")
    parser.add_argument("--target-rms", type=float, default=0.1,
                        help="Target RMS for normalization")
    parser.add_argument("--feat-scale", type=float, default=0.1,
                        help="Feature scale factor")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="ONNX Runtime thread count")
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed")
    parser.add_argument("--remove-long-sil", type=str2bool, default=False,
                        help="Remove long silences")
    parser.add_argument("--vocoder-path", type=str, default=None,
                        help="Path to vocoder checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_jieba_cache()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load runtime
    logging.info("Loading static ONNX models from %s ...", args.model_dir)
    runtime = StaticOnnxZipVoiceRuntime(
        model_dir=args.model_dir,
        num_threads=args.num_threads,
    )

    # Load tokenizer
    logging.info("Loading tokenizer ...")
    token_file = _PROJECT_ROOT / "models" / "tokens.txt"
    if not token_file.is_file():
        raise FileNotFoundError(f"tokens.txt not found at {token_file}")
    tokenizer = EmiliaTokenizer(token_file=str(token_file))

    # Load vocoder and feature extractor
    logging.info("Loading vocoder ...")
    vocoder = get_vocoder(vocos_local_path=args.vocoder_path)
    vocoder = vocoder.to(torch.device("cpu"))
    vocoder.eval()

    feature_extractor = VocosFbank()

    # Load prompt audio
    logging.info("Loading prompt audio: %s", args.prompt_wav)
    prompt_wav = load_prompt_wav(args.prompt_wav, sampling_rate=runtime.sampling_rate)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, args.target_rms)

    # Extract prompt features
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=runtime.sampling_rate
    )
    prompt_features = prompt_features.unsqueeze(0) * args.feat_scale  # [1, T_prompt, feat_dim]
    prompt_features_len = prompt_features.shape[1]

    # Tokenize
    logging.info("Tokenizing text ...")
    tokens = tokenizer.texts_to_token_ids([args.text])  # [[...]]
    prompt_tokens = tokenizer.texts_to_token_ids([args.prompt_text])  # [[...]]

    prompt_tokens_len = len(prompt_tokens[0])
    text_tokens_len = len(tokens[0])
    total_tokens_len = prompt_tokens_len + text_tokens_len + 1  # +1 for pad

    if total_tokens_len > runtime.max_tokens:
        logging.warning(
            "Total tokens %d exceeds max_tokens %d. Text will be truncated.",
            total_tokens_len, runtime.max_tokens,
        )
        # Truncate text tokens
        max_text_tokens = runtime.max_tokens - prompt_tokens_len - 1
        tokens[0] = tokens[0][:max_text_tokens]
        text_tokens_len = len(tokens[0])
        total_tokens_len = prompt_tokens_len + text_tokens_len + 1

    # Build cat_tokens: [prompt_tokens, text_tokens, pad_id]
    pad_id = tokenizer.pad_id
    cat_tokens_list = prompt_tokens[0] + tokens[0] + [pad_id]
    cat_tokens_np = np.zeros((1, runtime.max_tokens), dtype=np.int64)
    cat_tokens_np[0, :len(cat_tokens_list)] = cat_tokens_list
    # Fill remaining with pad_id
    cat_tokens_np[0, len(cat_tokens_list):] = pad_id

    logging.info(
        "Tokens: prompt=%d, text=%d, total(+pad)=%d, max_tokens=%d",
        prompt_tokens_len, text_tokens_len, total_tokens_len, runtime.max_tokens,
    )

    # Run sampling
    logging.info("Running ODE sampling (static ONNX) ...")
    t_start = time.perf_counter()

    pred_features, generated_frames = runtime.sample(
        cat_tokens=cat_tokens_np,
        prompt_tokens_len=prompt_tokens_len,
        text_tokens_len=text_tokens_len,
        prompt_features=prompt_features,
        prompt_features_len=prompt_features_len,
        speed=args.speed,
        t_shift=args.t_shift,
        guidance_scale=args.guidance_scale,
        num_step=args.num_step,
    )

    t_sampling = time.perf_counter() - t_start
    logging.info(
        "Sampling complete: %d generated frames, time=%.2fs",
        generated_frames, t_sampling,
    )

    # Vocoder
    logging.info("Running vocoder ...")
    t_voc_start = time.perf_counter()

    pred_features_t = torch.from_numpy(pred_features)
    pred_features_t = pred_features_t.permute(0, 2, 1) / args.feat_scale  # [1, feat_dim, T]

    wav = vocoder.decode(pred_features_t).squeeze(1).clamp(-1, 1)

    t_voc = time.perf_counter() - t_voc_start

    # Post-process
    if args.remove_long_sil:
        wav = remove_silence(wav, sample_rate=runtime.sampling_rate)

    # RMS normalization
    wav = rms_norm(wav, args.target_rms)[0]

    # Adjust volume
    if prompt_rms < args.target_rms:
        wav = wav * prompt_rms / args.target_rms

    # Save
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output_wav, wav.cpu(), sample_rate=runtime.sampling_rate)

    total_time = time.perf_counter() - t_start
    wav_seconds = wav.shape[-1] / runtime.sampling_rate
    rtf = total_time / wav_seconds if wav_seconds > 0 else float("inf")

    logging.info("=" * 60)
    logging.info(" Static ONNX Inference Result")
    logging.info("=" * 60)
    logging.info("  Output:           %s", args.output_wav)
    logging.info("  Generated frames: %d", generated_frames)
    logging.info("  Audio duration:   %.2fs", wav_seconds)
    logging.info("  Sampling time:    %.2fs", t_sampling)
    logging.info("  Vocoder time:     %.2fs", t_voc)
    logging.info("  Total time:       %.2fs", total_time)
    logging.info("  RTF:              %.2f", rtf)
    logging.info("  Sample rate:      %d", runtime.sampling_rate)
    logging.info("=" * 60)

    logging.info("Done! Output saved to %s", args.output_wav)


if __name__ == "__main__":
    main()
