#!/usr/bin/env python3
"""Generate richer calibration data for attn-loop decoder4 ZipVoice ONNX.

This script runs the same static encoder + four-part decoder path used by
decoder4 inference and records each ONNX input feed.  Unlike the earlier
decoder4 calibration flow, it records all ODE steps for every calibration
sample instead of only step 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_PYTHON_DIR = _PROJECT_ROOT / "python"
for path in [str(_PYTHON_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from onnx_zipvoice_decoder4_runtime import Decoder4StaticOnnxZipVoiceRuntime  # noqa: E402
from zipvoice.models.modules.solver import get_time_steps  # noqa: E402
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer  # noqa: E402
from zipvoice.utils.feature import VocosFbank  # noqa: E402
from zipvoice.utils.infer import load_prompt_wav, rms_norm  # noqa: E402


DEFAULT_TEXTS = [
    "你好，我们现在测试 ZipVoice 四段 decoder 的板端量化校准。",
    "这是一段包含较长中文内容的语音合成测试，用来覆盖不同的音素和停顿。",
    "ZipVoice uses flow matching to generate acoustic features for text to speech.",
    "Please calibrate the model with several time steps, different text lengths, and stable masks.",
]


@dataclass
class PromptSpec:
    wav: Path
    text: str


@dataclass
class RecordedFeed:
    model_key: str
    feed_index: int
    input_name: str
    value: np.ndarray


class RecordingSession:
    """Wrap an ONNX Runtime session and record every input feed."""

    def __init__(self, model_key: str, session: Any, collector: list[RecordedFeed]):
        self._model_key = model_key
        self._session = session
        self._collector = collector
        self._run_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def run(self, output_names, input_feed, *args, **kwargs):
        feed_index = self._run_count
        self._run_count += 1
        for input_name, value in input_feed.items():
            self._collector.append(
                RecordedFeed(
                    model_key=self._model_key,
                    feed_index=feed_index,
                    input_name=str(input_name),
                    value=np.asarray(value).copy(),
                )
            )
        return self._session.run(output_names, input_feed, *args, **kwargs)


def setup_jieba_cache() -> None:
    import jieba

    cache_dir = _PROJECT_ROOT / ".work_tmp" / "jieba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(cache_dir)
    jieba.dt.cache_file = "jieba.cache"


def _safe_name(input_name: str) -> str:
    return input_name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _normalize_for_pulsar(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape == ():
        return value.reshape(1)
    return value


def _write_feed_archives(out_dir: Path, feeds: list[RecordedFeed]) -> dict:
    by_model_input: dict[tuple[str, str], list[RecordedFeed]] = defaultdict(list)
    for feed in feeds:
        by_model_input[(feed.model_key, feed.input_name)].append(feed)

    archives: dict = {}
    for (model_key, input_name), records in sorted(by_model_input.items()):
        safe_name = _safe_name(input_name)
        input_dir = out_dir / model_key / safe_name
        input_dir.mkdir(parents=True, exist_ok=True)
        tar_path = out_dir / model_key / f"{safe_name}.tar.gz"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        with tarfile.open(tar_path, "w:gz") as tar:
            for sample_index, record in enumerate(records):
                value = _normalize_for_pulsar(record.value)
                npy_path = input_dir / f"{sample_index:05d}.npy"
                np.save(npy_path, value)
                tar.add(npy_path, arcname=f"{safe_name}/{npy_path.name}")
                entries.append(
                    {
                        "file": str(npy_path.relative_to(out_dir)),
                        "feed_index": int(record.feed_index),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
        archives.setdefault(model_key, {})[input_name] = {
            "tar": str(tar_path.relative_to(out_dir)),
            "count": len(records),
            "entries": entries,
        }
    return archives


def _parse_speeds(text: str) -> list[float]:
    speeds = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        speeds.append(float(item))
    if not speeds:
        raise ValueError("--speed-list must contain at least one speed")
    return speeds


def _load_prompt_specs(args: argparse.Namespace) -> list[PromptSpec]:
    wavs = [Path(value).expanduser().resolve() for value in args.prompt_wav]
    texts = list(args.prompt_text)
    if len(texts) == 1 and len(wavs) > 1:
        texts = texts * len(wavs)
    if len(wavs) != len(texts):
        raise ValueError(
            f"--prompt-wav count ({len(wavs)}) must match --prompt-text count "
            f"({len(texts)}), or provide one prompt text to reuse"
        )
    specs = [PromptSpec(wav=wav, text=text) for wav, text in zip(wavs, texts)]
    for spec in specs:
        if not spec.wav.is_file():
            raise FileNotFoundError(spec.wav)
    return specs


def _install_recorders(
    runtime: Decoder4StaticOnnxZipVoiceRuntime,
    collector: list[RecordedFeed],
) -> None:
    runtime.encoder_session = RecordingSession(
        "encoder_core", runtime.encoder_session, collector
    )
    runtime.part_sessions = [
        RecordingSession(f"fm_decoder_part{index}", session, collector)
        for index, session in enumerate(runtime.part_sessions)
    ]


def _sample_and_record(
    runtime: Decoder4StaticOnnxZipVoiceRuntime,
    cat_tokens: np.ndarray,
    prompt_tokens_len: int,
    text_tokens_len: int,
    prompt_features: torch.Tensor,
    prompt_features_len: int,
    speed: float,
    t_shift: float,
    guidance_scale: float,
    num_step: int,
    seed: int,
) -> dict[str, Any]:
    encoded = runtime.run_encoder(cat_tokens)
    text_condition, features_len = runtime.duration_expand(
        encoded,
        prompt_tokens_len,
        text_tokens_len,
        prompt_features_len,
        speed,
    )

    seq_len = runtime.decoder_seq_len or features_len
    if features_len > seq_len:
        raise ValueError(f"features_len={features_len} exceeds decoder seq_len={seq_len}")
    if prompt_features.shape[1] > seq_len:
        raise ValueError(
            f"prompt_features_len={prompt_features.shape[1]} exceeds decoder seq_len={seq_len}"
        )

    text_cond_padded = np.zeros((1, seq_len, runtime.feat_dim), dtype=np.float32)
    text_cond_padded[0, :features_len] = text_condition[0, :features_len]

    prompt_np = prompt_features.numpy().astype(np.float32)
    speech_cond_padded = np.zeros((1, seq_len, runtime.feat_dim), dtype=np.float32)
    speech_cond_padded[0, : prompt_np.shape[1]] = prompt_np[0]

    rng = np.random.RandomState(seed)
    x = rng.randn(1, seq_len, runtime.feat_dim).astype(np.float32)
    x[:, features_len:, :] = 0.0

    padding_mask = np.zeros((1, seq_len), dtype=np.bool_)
    padding_mask[:, features_len:] = True
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )
    gs = np.array([guidance_scale], dtype=np.float32)

    for step in range(num_step):
        t_val = np.array([float(timesteps[step])], dtype=np.float32)
        v = runtime.run_decoder(
            t_val,
            x,
            text_cond_padded,
            speech_cond_padded,
            gs,
            padding_mask,
        )
        dt = float(timesteps[step + 1] - timesteps[step])
        x = (x + v * dt).astype(np.float32)
        x[:, features_len:, :] = 0.0

    generated_frames = features_len - prompt_features_len
    if generated_frames <= 0:
        generated_frames = features_len

    return {
        "features_len": int(features_len),
        "generated_frames": int(generated_frames),
        "seq_len": int(seq_len),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rich calibration data for ZipVoice decoder4 ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="models/decoder4_static_distill_1024_ax630c",
        help="Directory containing encoder_core.onnx and fm_decoder_part*.onnx",
    )
    parser.add_argument(
        "--output-dir",
        default="model_convert/calib_data_decoder4_rich_distill_1024",
        help="Output calibration directory",
    )
    parser.add_argument(
        "--prompt-wav",
        action="append",
        required=True,
        help="Prompt wav path. Can be repeated.",
    )
    parser.add_argument(
        "--prompt-text",
        action="append",
        required=True,
        help="Prompt transcription. Repeat to match --prompt-wav, or provide one to reuse.",
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Calibration text. Can be repeated; defaults to a mixed built-in set.",
    )
    parser.add_argument(
        "--num-calib-texts",
        type=int,
        default=None,
        help="Number of built-in texts to use when --text is not supplied.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--t-shift", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--speed-list",
        default="1.0,0.85,1.15,1.0",
        help="Comma-separated speed values, cycled across samples.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--target-rms", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_jieba_cache()

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_specs = _load_prompt_specs(args)
    speeds = _parse_speeds(args.speed_list)
    texts = args.text if args.text else DEFAULT_TEXTS
    if args.num_calib_texts is not None and not args.text:
        texts = texts[: args.num_calib_texts]

    logging.info("Loading decoder4 static ONNX runtime from %s", model_dir)
    runtime = Decoder4StaticOnnxZipVoiceRuntime(
        model_dir=str(model_dir),
        num_threads=args.num_threads,
    )
    collector: list[RecordedFeed] = []
    _install_recorders(runtime, collector)

    token_file = _PROJECT_ROOT / "models" / "tokens.txt"
    tokenizer = EmiliaTokenizer(token_file=str(token_file))
    feature_extractor = VocosFbank()
    pad_id = tokenizer.pad_id

    samples = []
    sample_index = 0
    for prompt_index, prompt in enumerate(prompt_specs):
        logging.info("Loading prompt %d/%d: %s", prompt_index + 1, len(prompt_specs), prompt.wav)
        prompt_wav = load_prompt_wav(str(prompt.wav), sampling_rate=runtime.sampling_rate)
        prompt_wav, prompt_rms = rms_norm(prompt_wav, args.target_rms)
        prompt_features = feature_extractor.extract(
            prompt_wav,
            sampling_rate=runtime.sampling_rate,
        )
        prompt_features = prompt_features.unsqueeze(0) * args.feat_scale
        prompt_features_len = int(prompt_features.shape[1])
        if prompt_features_len > runtime.max_feat_len:
            raise ValueError(
                f"prompt {prompt.wav} has {prompt_features_len} frames, exceeding "
                f"max_feat_len={runtime.max_feat_len}"
            )

        prompt_tokens = tokenizer.texts_to_token_ids([prompt.text])[0]
        prompt_tokens_len = len(prompt_tokens)

        for text in texts:
            speed = speeds[sample_index % len(speeds)]
            sample_seed = args.seed + sample_index
            before_count = len(collector)
            text_tokens = tokenizer.texts_to_token_ids([text])[0]
            text_tokens_len = len(text_tokens)

            max_text_tokens = runtime.max_tokens - prompt_tokens_len - 1
            if max_text_tokens < 1:
                raise ValueError(
                    f"prompt {prompt.wav} token length {prompt_tokens_len} leaves no room "
                    f"with max_tokens={runtime.max_tokens}"
                )
            if text_tokens_len > max_text_tokens:
                logging.warning(
                    "Truncating text tokens from %d to %d for sample %d",
                    text_tokens_len,
                    max_text_tokens,
                    sample_index,
                )
                text_tokens = text_tokens[:max_text_tokens]
                text_tokens_len = len(text_tokens)

            cat_tokens = prompt_tokens + text_tokens + [pad_id]
            cat_tokens_np = np.full((1, runtime.max_tokens), pad_id, dtype=np.int64)
            cat_tokens_np[0, : len(cat_tokens)] = np.asarray(cat_tokens, dtype=np.int64)

            logging.info(
                "Sample %d: prompt=%s text_tokens=%d prompt_frames=%d speed=%.2f seed=%d",
                sample_index,
                prompt.wav.name,
                text_tokens_len,
                prompt_features_len,
                speed,
                sample_seed,
            )
            metrics = _sample_and_record(
                runtime=runtime,
                cat_tokens=cat_tokens_np,
                prompt_tokens_len=prompt_tokens_len,
                text_tokens_len=text_tokens_len,
                prompt_features=prompt_features,
                prompt_features_len=prompt_features_len,
                speed=speed,
                t_shift=args.t_shift,
                guidance_scale=args.guidance_scale,
                num_step=args.num_step,
                seed=sample_seed,
            )
            samples.append(
                {
                    "index": sample_index,
                    "prompt_index": prompt_index,
                    "prompt_wav": str(prompt.wav),
                    "prompt_text": prompt.text,
                    "prompt_tokens_len": int(prompt_tokens_len),
                    "prompt_features_len": int(prompt_features_len),
                    "prompt_rms": float(prompt_rms),
                    "text": text,
                    "text_tokens_len": int(text_tokens_len),
                    "speed": float(speed),
                    "seed": int(sample_seed),
                    "recorded_feed_count": int(len(collector) - before_count),
                    **metrics,
                }
            )
            sample_index += 1

    archives = _write_feed_archives(out_dir, collector)
    manifest = {
        "model_dir": str(model_dir),
        "output_dir": str(out_dir),
        "num_step": args.num_step,
        "t_shift": args.t_shift,
        "guidance_scale": args.guidance_scale,
        "speed_list": speeds,
        "feat_scale": args.feat_scale,
        "target_rms": args.target_rms,
        "feed_count": len(collector),
        "sample_count": len(samples),
        "samples": samples,
        "archives": archives,
    }
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("Calibration data written to %s", out_dir)
    logging.info("Total samples: %d", len(samples))
    logging.info("Total recorded feeds: %d", len(collector))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    main()
