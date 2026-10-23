#!/usr/bin/env python3
"""
ZipVoice static ONNX inference runtime.

Pipeline:
  encoder_core(static) → host duration expansion → fm_decoder(static) → vocoder(PyTorch)

The text encoder is split:
  1. encoder_core.onnx: embed + text_encoder → encoded tokens
  2. Host-side duration expansion to match feature length
  3. fm_decoder.onnx: flow-matching ODE solver (Euler, 16 steps)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from zipvoice.models.modules.solver import get_time_steps


class StaticOnnxZipVoiceRuntime:
    """Runs ZipVoice inference using static-shape ONNX models."""

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 4,
        encoder_model_path: Optional[str] = None,
    ):
        """
        Args:
            model_dir: Directory containing encoder_core.onnx, fm_decoder.onnx,
                       and runtime_config.json
            num_threads: Number of threads for ONNX Runtime
            encoder_model_path: Optional override for encoder_core.onnx.  The
                       decoder and runtime_config are still loaded from model_dir.
        """
        self.model_dir = Path(model_dir)
        self.encoder_model_path = Path(encoder_model_path) if encoder_model_path else None
        self._load_config()
        self._init_sessions(num_threads)

    def _load_config(self):
        config_path = self.model_dir / "runtime_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        self.max_tokens = config.get("max_tokens", 384)
        self.max_feat_len = config.get("max_feat_len", 2048)
        self.feat_dim = config.get("feat_dim", 100)
        self.sampling_rate = config.get("sampling_rate", 24000)
        logging.info(
            "Runtime config: max_tokens=%d, max_feat_len=%d, feat_dim=%d",
            self.max_tokens, self.max_feat_len, self.feat_dim,
        )

    def _init_sessions(self, num_threads: int):
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = num_threads
        sess_opts.intra_op_num_threads = num_threads
        # Graph optimizations are safe now that the SimpleDownsample
        # zero-padding Expand bug has been fixed at the PyTorch level.
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session_opts = sess_opts

        encoder_path = self.encoder_model_path or (self.model_dir / "encoder_core.onnx")
        decoder_path = self.model_dir / "fm_decoder.onnx"

        if not encoder_path.exists():
            raise FileNotFoundError(f"encoder_core.onnx not found at {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"fm_decoder.onnx not found at {decoder_path}")

        self.encoder_path = encoder_path
        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        decoder_inputs = self.decoder_session.get_inputs()
        self.decoder_has_padding_mask = len(decoder_inputs) >= 6
        decoder_x_shape = decoder_inputs[1].shape
        self.decoder_seq_len = decoder_x_shape[1] if isinstance(decoder_x_shape[1], int) else None
        logging.info("ONNX sessions initialized successfully")

    @property
    def sessions(self) -> Dict[str, ort.InferenceSession]:
        """Return all sessions for calibration recording."""
        return {
            "encoder_core": self.encoder_session,
            "fm_decoder": self.decoder_session,
        }

    def run_encoder(self, cat_tokens: np.ndarray) -> np.ndarray:
        """Run the encoder core model.

        Args:
            cat_tokens: [1, max_tokens] int64 array (prompt + text + pad, padded)

        Returns:
            encoded: [1, max_tokens, feat_dim] float32 array
        """
        out = self.encoder_session.run(
            [self.encoder_session.get_outputs()[0].name],
            {self.encoder_session.get_inputs()[0].name: cat_tokens},
        )
        return out[0]

    def run_decoder(
        self,
        t: np.ndarray,
        x: np.ndarray,
        text_condition: np.ndarray,
        speech_condition: np.ndarray,
        guidance_scale: np.ndarray,
        padding_mask: np.ndarray = None,
    ) -> np.ndarray:
        """Run the fm_decoder model.

        Args:
            t: [1] float32
            x: [1, seq_len, feat_dim] float32
            text_condition: [1, seq_len, feat_dim] float32
            speech_condition: [1, seq_len, feat_dim] float32
            guidance_scale: [1] float32
            padding_mask: [1, seq_len] bool; True = pad position

        Returns:
            v: [1, seq_len, feat_dim] float32 (velocity)
        """
        inputs = self.decoder_session.get_inputs()
        outputs = self.decoder_session.get_outputs()
        seq_len = x.shape[1]
        feed = {
            inputs[0].name: t,
            inputs[1].name: x,
            inputs[2].name: text_condition,
            inputs[3].name: speech_condition,
            inputs[4].name: guidance_scale,
        }
        # If model expects padding_mask (6 inputs), include it
        if len(inputs) >= 6:
            feed[inputs[5].name] = padding_mask if padding_mask is not None \
                else np.zeros((1, seq_len), dtype=np.bool_)
        out = self.decoder_session.run([outputs[0].name], feed)
        return out[0]

    def duration_expand(
        self,
        encoded: np.ndarray,
        prompt_tokens_len: int,
        text_tokens_len: int,
        prompt_features_len: int,
        speed: float,
    ) -> Tuple[np.ndarray, int]:
        """Expand encoded tokens to feature-length text_condition.

        This replicates the OnnxTextModel's duration expansion logic on the host.

        Args:
            encoded: [1, max_tokens, feat_dim] from encoder_core
            prompt_tokens_len: number of prompt tokens
            text_tokens_len: number of text tokens
            prompt_features_len: number of prompt feature frames
            speed: speed control factor

        Returns:
            text_condition: [1, actual_feat_len, feat_dim]
            actual_feat_len: actual feature length
        """
        total_tokens_len = prompt_tokens_len + text_tokens_len

        # Compute feature length (matching OnnxTextModel logic)
        features_len = int(np.ceil(
            prompt_features_len / prompt_tokens_len * total_tokens_len / speed
        ))

        if features_len > self.max_feat_len:
            logging.warning(
                "Computed features_len=%d exceeds max_feat_len=%d, clamping",
                features_len, self.max_feat_len,
            )
            features_len = self.max_feat_len

        # token_dur = floor(features_len / total_tokens_len)
        token_dur = features_len // total_tokens_len

        # Remove the last (pad) position from encoded
        # encoded is [1, max_tokens, feat_dim], we use first total_tokens_len positions
        embed_no_pad = encoded[0, :total_tokens_len, :]  # [total_tokens_len, feat_dim]

        # Expand each token by token_dur and reshape
        text_condition = np.repeat(embed_no_pad, token_dur, axis=0)  # [total_tokens_len * token_dur, feat_dim]

        # Pad to features_len with the pad token's embedding (position total_tokens_len,
        # which is the first pad token after prompt+text tokens).
        # This matches the OnnxTextModel which uses embed[:, -1:, :] (the trailing
        # pad token) for the residual frames.
        residual = features_len - text_condition.shape[0]
        if residual > 0:
            last_embed = encoded[0, total_tokens_len:total_tokens_len + 1, :]
            pad_part = np.repeat(last_embed, residual, axis=0)
            text_condition = np.concatenate([text_condition, pad_part], axis=0)

        # Clip to features_len
        text_condition = text_condition[:features_len, :]

        return text_condition[np.newaxis, :, :], features_len

    def sample(
        self,
        cat_tokens: np.ndarray,
        prompt_tokens_len: int,
        text_tokens_len: int,
        prompt_features: Tensor,
        prompt_features_len: int,
        speed: float = 1.0,
        t_shift: float = 0.5,
        guidance_scale: float = 1.0,
        num_step: int = 16,
    ) -> Tuple[np.ndarray, int]:
        """Run the full sampling pipeline.

        Args:
            cat_tokens: [1, max_tokens] int64 — prompt+text tokens, padded
            prompt_tokens_len: actual number of prompt tokens
            text_tokens_len: actual number of text tokens
            prompt_features: [1, prompt_frames, feat_dim] PyTorch tensor
            prompt_features_len: number of prompt feature frames
            speed: speed control
            t_shift: time shift for ODE
            guidance_scale: classifier-free guidance scale
            num_step: number of ODE steps

        Returns:
            pred_features: [1, generated_frames, feat_dim] — generated features (trimmed)
            generated_frames: actual number of generated feature frames
        """
        # Step 1: Run encoder
        encoded = self.run_encoder(cat_tokens)  # [1, max_tokens, feat_dim]

        # Step 2: Duration expansion
        text_condition, features_len = self.duration_expand(
            encoded, prompt_tokens_len, text_tokens_len,
            prompt_features_len, speed,
        )

        # Step 3: Prepare inputs for ODE solver. Prefer exact-length decoding when
        # the ONNX keeps a dynamic sequence axis; otherwise pad to the decoder's
        # fixed input length.
        seq_len = self.decoder_seq_len or features_len
        if features_len > seq_len:
            raise ValueError(
                f"features_len={features_len} exceeds decoder sequence length {seq_len}"
            )
        if (
            self.decoder_seq_len is not None
            and not self.decoder_has_padding_mask
            and features_len != seq_len
        ):
            raise ValueError(
                "Fixed no-mask decoder requires exact feature length: "
                f"features_len={features_len}, decoder_seq_len={seq_len}. "
                "Re-export fm_decoder with MAX_FEAT_LEN equal to this features_len."
            )
        text_cond_padded = np.zeros((1, seq_len, self.feat_dim), dtype=np.float32)
        text_cond_padded[0, :features_len, :] = text_condition[0, :features_len, :]

        prompt_np = prompt_features.numpy().astype(np.float32)
        prompt_actual_len = prompt_np.shape[1]
        speech_cond_padded = np.zeros((1, seq_len, self.feat_dim), dtype=np.float32)
        speech_cond_padded[0, :prompt_actual_len, :] = prompt_np[0, :, :]

        # Initialize x as random noise. Padded frames must stay zero; otherwise
        # ZipFormer's downsample blocks can mix padding noise into real frames
        # before the padding mask is applied inside attention.
        rng = np.random.RandomState()
        x = rng.randn(1, seq_len, self.feat_dim).astype(np.float32)
        if features_len < seq_len:
            x[:, features_len:, :] = 0.0

        # ODE timesteps
        timesteps = get_time_steps(
            t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift,
        )
        gs = np.array([guidance_scale], dtype=np.float32)

        # Step 4: ODE Euler integration
        padding_mask = np.zeros((1, seq_len), dtype=np.bool_)
        padding_mask[:, features_len:] = True

        for step in range(num_step):
            t_val = np.array([float(timesteps[step])], dtype=np.float32)
            v = self.run_decoder(t_val, x, text_cond_padded, speech_cond_padded, gs, padding_mask)
            dt = float(timesteps[step + 1] - timesteps[step])
            x = x + v * dt
            if features_len < seq_len:
                x[:, features_len:, :] = 0.0

        # Step 5: Trim output to actual feature length (excluding prompt portion)
        generated_frames = features_len - prompt_features_len
        if generated_frames <= 0:
            generated_frames = features_len
            pred_features = x[0, :features_len, :]
        else:
            pred_features = x[0, prompt_features_len:features_len, :]

        return pred_features[np.newaxis, :, :], generated_frames
