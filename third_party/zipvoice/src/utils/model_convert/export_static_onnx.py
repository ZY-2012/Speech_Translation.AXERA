#!/usr/bin/env python3
"""
Export static-shape ONNX models for ZipVoice.

Pipeline:
  PyTorch checkpoint → static encoder_core.onnx + fm_decoder.onnx

The text encoder is split into two parts:
  1. encoder_core.onnx: embed + text_encoder (no duration expansion)
  2. Duration expansion: done on host (Python/C++)

The fm_decoder.onnx runs the flow-matching decoder.

Key fix: encoder_pos modules are replaced with trace-friendly
versions BEFORE convert_scaled_to_non_scaled, avoiding
RecursiveScriptModule/If nodes. This makes onnxsim work without
segfaults.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import onnx
import safetensors.torch
import torch
from torch import Tensor, nn

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.scaling_converter import convert_scaled_to_non_scaled


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)


import math


class _TraceFriendlyPosEncoding(nn.Module):
    """Drop-in replacement for CompactRelPositionalEncoding that traces cleanly.

    This is a simplified re-implementation that does not cache self.pe
    across calls — it recomputes the positional encoding every forward().
    State-less forward avoids the tracer seeing mutations that cause
    Tensor-valued Constant node differences.

    The original module itself would trace fine, but convert_scaled_to_non_scaled
    wraps it with torch.jit.script(), producing RecursiveScriptModule → If nodes
    in ONNX.  By replacing it with a different class BEFORE conversion, the
    converter leaves it alone and jit.trace handles it normally.
    """

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.embed_dim = orig.embed_dim
        self.length_factor = orig.length_factor
        self.dropout = orig.dropout

    def _compute_pe(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        """Compute positional encoding for the given input shape (no caching)."""
        T = x.size(0) + left_context_len

        arange = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)
        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        compression_length = self.embed_dim ** 0.5
        x_compressed = (
            compression_length
            * arange.sign()
            * ((arange.abs() + compression_length).log() - math.log(compression_length))
        )
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)
        x_atan = (x_compressed / length_scale).atan()
        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(arange.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0

        return pe

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        pe = self._compute_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        T = x.size(0)
        pe_slice = pe[pe.size(0) // 2 - x_size_left + 1 : pe.size(0) // 2 + T, :]
        pos_emb = pe_slice.unsqueeze(0)
        return self.dropout(pos_emb)


def _replace_encoder_pos(model: nn.Module) -> None:
    """Replace all CompactRelPositionalEncoding with _TraceFriendlyPosEncoding.

    This must happen BEFORE convert_scaled_to_non_scaled, which otherwise
    wraps CompactRelPositionalEncoding with torch.jit.script() and causes
    If nodes in the ONNX graph (which segfault onnxsim).
    """
    count = 0
    for module_path, module in list(model.named_modules()):
        if not module_path.endswith(".encoder_pos"):
            continue
        if not isinstance(module, (nn.Module,)):
            continue
        # Only replace actual CompactRelPositionalEncoding, not already-replaced
        cls_name = type(module).__name__
        if cls_name != "CompactRelPositionalEncoding":
            continue

        replacement = _TraceFriendlyPosEncoding(module)
        logging.info(
            "Replacing %s (%s, embed_dim=%d) with trace-friendly version",
            module_path, cls_name, module.embed_dim,
        )

        *parent_parts, attr = module_path.split(".")
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)
        setattr(parent, attr, replacement)
        count += 1

    logging.info("Replaced %d CompactRelPositionalEncoding module(s)", count)


def _patch_simple_downsample(model: nn.Module) -> None:
    """Patch SimpleDownsample.forward to skip zero-padding Expand+Concat.

    ONNX Runtime's Expand does not handle zero-sized output correctly
    (shape [0, B, D] becomes [1, B, D]), causing a +1 offset in the
    sequence dimension at every Concat.  Bypassing the pad path when
    pad==0 produces a clean ONNX graph without this bug.
    """
    import types

    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name != "SimpleDownsample":
            continue

        ds = module.downsample

        def _patched_forward(self, src):
            (seq_len, batch_size, in_channels) = src.shape
            ds2 = self.downsample
            d_seq_len = (seq_len + ds2 - 1) // ds2
            pad = d_seq_len * ds2 - seq_len
            if pad == 0:
                # Skip Expand + Concat — shape is already a multiple of ds
                src_reshaped = src.reshape(d_seq_len, ds2, batch_size, in_channels)
            else:
                src_extra = src[src.shape[0] - 1:].expand(pad, src.shape[1], src.shape[2])
                src_padded = torch.cat((src, src_extra), dim=0)
                src_reshaped = src_padded.reshape(d_seq_len, ds2, batch_size, in_channels)
            weights = self.bias.softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
            return (src_reshaped * weights).sum(dim=1)

        module.forward = types.MethodType(_patched_forward, module)
        logging.info("Patched SimpleDownsample (ds=%d) to skip zero-pad Expand+Concat", ds)


def _build_onnxsim_input_shapes(model: onnx.ModelProto) -> Dict[str, List[int]]:
    meta = {item.key: item.value for item in model.metadata_props}
    seq_len = int(meta.get("max_feat_len", "2048"))
    input_shapes: Dict[str, List[int]] = {}

    for value_info in model.graph.input:
        dims: List[int] = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                dims.append(dim.dim_value)
            elif dim.dim_param == "seq_len":
                dims.append(seq_len)
            else:
                dims.append(1)
        if dims:
            input_shapes[value_info.name] = dims

    return input_shapes


def _has_dynamic_input_shape(model: onnx.ModelProto) -> bool:
    for value_info in model.graph.input:
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                return True
    return False


def _run_onnxsim(model: onnx.ModelProto, name: str) -> onnx.ModelProto:
    """Run onnxsim with onnxslim fallback.

    Tries onnxsim first. If onnxsim check fails (returns ok=False),
    falls back to onnxslim. If onnxslim also fails, returns the original
    model with a warning — the caller should then try re-exporting with
    modified model structure.
    """
    # Try onnxsim first
    try:
        from onnxsim import simplify
        logging.info("Running onnxsim on %s ...", name)
        simplify_kwargs = {
            "check_n": 0,
            "perform_optimization": True,
        }
        if _has_dynamic_input_shape(model):
            simplify_kwargs["dynamic_input_shape"] = True
            simplify_kwargs["input_shapes"] = _build_onnxsim_input_shapes(model)
        else:
            simplify_kwargs["input_shapes"] = _build_onnxsim_input_shapes(model)
        optimized, ok = simplify(
            model,
            **simplify_kwargs,
        )
        if ok:
            logging.info("onnxsim OK for %s (nodes: %d → %d, %.1fMB → %.1fMB)",
                         name, len(model.graph.node), len(optimized.graph.node),
                         model.ByteSize()/1024/1024, optimized.ByteSize()/1024/1024)
            return optimized
        logging.warning("onnxsim check returned ok=False for %s (nodes=%d), trying onnxslim...",
                       name, len(model.graph.node))
    except Exception as exc:
        logging.warning("onnxsim exception for %s (%s), trying onnxslim...", name, exc)

    # Fallback: onnxslim
    try:
        from onnxslim import slim
        optimized = slim(model, size_threshold=10 * 1024 * 1024)
        logging.info("onnxslim OK for %s (nodes: %d → %d)",
                     name, len(model.graph.node), len(optimized.graph.node))
        return optimized
    except ImportError:
        logging.warning("onnxslim not installed for %s", name)
    except Exception as exc2:
        logging.warning("onnxslim also failed for %s: %s", name, exc2)

    # If both onnxsim and onnxslim failed, return the original model
    logging.warning("Both onnxsim and onnxslim failed for %s — keeping unoptimized model", name)
    return model


def _run_onnxsim_with_retry(
    model: onnx.ModelProto,
    name: str,
    export_fn,
    max_retries: int = 2,
) -> onnx.ModelProto:
    """Run onnxsim with retry: if it fails, try modifying model structure.

    Args:
        model: The ONNX model to simplify.
        name: Model name for logging.
        export_fn: A callable that re-exports the model after structural changes.
                   Called as export_fn() and should return a new onnx.ModelProto.
        max_retries: Maximum number of retry attempts.

    Returns:
        The optimized model, or the original if all attempts fail.
    """
    result = _run_onnxsim(model, name)
    # Check if optimization actually changed the model
    if len(result.graph.node) < len(model.graph.node) * 0.99:
        return result
    # onnxsim didn't significantly reduce nodes — model might be fine as-is
    if len(result.graph.node) == len(model.graph.node):
        return result

    # onnxsim returned original model (failed) — try re-exporting with patches
    logging.warning("onnxsim did not optimize %s — attempting structural fixes...", name)
    for attempt in range(1, max_retries + 1):
        logging.info("Retry %d/%d: re-exporting %s with structural changes...", attempt, max_retries, name)
        try:
            new_model = export_fn()
            result = _run_onnxsim(new_model, name)
            if len(result.graph.node) < len(model.graph.node) * 0.99:
                logging.info("Retry %d succeeded for %s", attempt, name)
                return result
        except Exception as exc:
            logging.error("Retry %d failed for %s: %s", attempt, name, exc)

    logging.warning("All retries failed for %s — keeping unoptimized model", name)
    return model


class EncoderCoreWrapper(nn.Module):
    """Wrapper for embed + text_encoder only (no duration expansion)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.embed = model.embed
        self.text_encoder = model.text_encoder
        self.pad_id = model.pad_id

    def forward(self, cat_tokens: Tensor) -> Tensor:
        # Use token equality to pad_id as padding_mask, not position-based.
        # This correctly masks ALL pad tokens in the fixed-size static input.
        padding_mask = (cat_tokens == self.pad_id)
        embed = self.embed(cat_tokens)
        encoded = self.text_encoder(x=embed, t=None, padding_mask=padding_mask)
        return encoded


class FmDecoderWrapper(nn.Module):
    """Wrapper for the flow-matching decoder with padding_mask support.

    When inputs are padded to fixed max_feat_len, the padding_mask prevents
    padded positions from corrupting attention in the zipformer encoders.
    """

    def __init__(self, model: nn.Module, distill: bool = False):
        super().__init__()
        self.distill = distill
        self.fm_decoder = model.fm_decoder
        self._model = model  # keep reference for forward_fm_decoder_with_mask
        self.feat_dim = model.feat_dim

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        guidance_scale: Tensor,
        padding_mask: Tensor,  # [1, T] boolean; True = pad position
    ) -> Tensor:
        if self.distill:
            return self._model.forward_fm_decoder(
                t=t, xt=x, text_condition=text_condition,
                speech_condition=speech_condition, guidance_scale=guidance_scale,
                padding_mask=padding_mask,
            )
        x2 = torch.cat([x, x], dim=0)
        mask2 = torch.cat([padding_mask, padding_mask], dim=0)
        text_cond2 = torch.cat(
            [torch.zeros_like(text_condition), text_condition], dim=0
        )
        speech_cond2 = torch.cat(
            [
                torch.where(t > 0.5, torch.zeros_like(speech_condition), speech_condition),
                speech_condition,
            ],
            dim=0,
        )
        t2 = torch.cat([t.reshape(1, 1, 1), t.reshape(1, 1, 1)], dim=0)
        guidance_scale = torch.where(t > 0.5, guidance_scale, guidance_scale * 2.0)
        data_uncond, data_cond = self._model.forward_fm_decoder(
            t=t2, xt=x2, text_condition=text_cond2,
            speech_condition=speech_cond2, padding_mask=mask2,
        ).chunk(2, dim=0)
        v = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
        return v


class FmDecoderWrapperNoMask(nn.Module):
    """Flow-matching decoder wrapper matching the original ONNX export path."""

    def __init__(self, model: nn.Module, distill: bool = False):
        super().__init__()
        self.distill = distill
        self.fm_decoder = model.fm_decoder
        self._model = model
        self.feat_dim = model.feat_dim

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        if self.distill:
            return self._model.forward_fm_decoder(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                guidance_scale=guidance_scale,
            )
        x2 = torch.cat([x, x], dim=0)
        text_cond2 = torch.cat(
            [torch.zeros_like(text_condition), text_condition], dim=0
        )
        speech_cond2 = torch.cat(
            [
                torch.where(t > 0.5, torch.zeros_like(speech_condition), speech_condition),
                speech_condition,
            ],
            dim=0,
        )
        t2 = torch.cat([t.reshape(1, 1, 1), t.reshape(1, 1, 1)], dim=0)
        guidance_scale = torch.where(t > 0.5, guidance_scale, guidance_scale * 2.0)
        data_uncond, data_cond = self._model.forward_fm_decoder(
            t=t2,
            xt=x2,
            text_condition=text_cond2,
            speech_condition=speech_cond2,
        ).chunk(2, dim=0)
        return (1 + guidance_scale) * data_cond - guidance_scale * data_uncond


def export_encoder_core(
    model: EncoderCoreWrapper,
    filename: str,
    max_tokens: int,
    feat_dim: int,
    opset_version: int = 13,
) -> None:
    cat_tokens = torch.zeros((1, max_tokens), dtype=torch.int64)
    cat_tokens[:, :min(10, max_tokens)] = torch.arange(1, min(11, max_tokens + 1))

    logging.info("Tracing encoder_core with max_tokens=%d ...", max_tokens)
    traced = torch.jit.trace(model, (cat_tokens,))

    torch.onnx.export(
        traced, (cat_tokens,), filename,
        verbose=False, opset_version=opset_version,
        input_names=["cat_tokens"], output_names=["encoded"],
        dynamic_axes={},
    )

    meta_data = {
        "version": "1", "model_author": "k2-fsa",
        "comment": "ZipVoice encoder core (static)",
        "max_tokens": str(max_tokens), "feat_dim": str(feat_dim),
    }
    add_meta_data(filename=filename, meta_data=meta_data)
    logging.info("Exported encoder_core to %s", filename)


def export_fm_decoder(
    model: nn.Module,
    filename: str,
    max_feat_len: int,
    opset_version: int = 13,
    dynamic_seq_len: bool = False,
) -> None:
    feat_dim = model.feat_dim
    t = torch.tensor(0.5, dtype=torch.float32)
    x = torch.randn(1, max_feat_len, feat_dim, dtype=torch.float32)
    text_condition = torch.randn(1, max_feat_len, feat_dim, dtype=torch.float32)
    speech_condition = torch.randn(1, max_feat_len, feat_dim, dtype=torch.float32)
    guidance_scale = torch.tensor(1.0, dtype=torch.float32)
    logging.info("Tracing fm_decoder with max_feat_len=%d ...", max_feat_len)
    if isinstance(model, FmDecoderWrapperNoMask):
        traced = torch.jit.trace(
            model, (t, x, text_condition, speech_condition, guidance_scale)
        )
        torch.onnx.export(
            traced,
            (t, x, text_condition, speech_condition, guidance_scale),
            filename,
            verbose=False,
            opset_version=opset_version,
            input_names=["t", "x", "text_condition", "speech_condition", "guidance_scale"],
            output_names=["v"],
            dynamic_axes=(
                {
                    "x": {1: "seq_len"},
                    "text_condition": {1: "seq_len"},
                    "speech_condition": {1: "seq_len"},
                    "v": {1: "seq_len"},
                }
                if dynamic_seq_len else {}
            ),
        )
    else:
        # Half-padded input: first half real, second half padding
        padding_mask = torch.zeros(1, max_feat_len, dtype=torch.bool)
        padding_mask[:, max_feat_len // 2:] = True
        traced = torch.jit.trace(
            model, (t, x, text_condition, speech_condition, guidance_scale, padding_mask)
        )
        torch.onnx.export(
            traced,
            (t, x, text_condition, speech_condition, guidance_scale, padding_mask),
            filename,
            verbose=False,
            opset_version=opset_version,
            input_names=["t", "x", "text_condition", "speech_condition", "guidance_scale", "padding_mask"],
            output_names=["v"],
            dynamic_axes={},
        )

    meta_data = {
        "version": "1", "model_author": "k2-fsa",
        "comment": "ZipVoice flow-matching decoder (static)",
        "feat_dim": str(feat_dim), "max_feat_len": str(max_feat_len),
        "sample_rate": "24000", "n_fft": "1024",
        "hop_length": "256", "num_mels": "100",
    }
    add_meta_data(filename=filename, meta_data=meta_data)
    logging.info("Exported fm_decoder to %s", filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export static-shape ONNX models for ZipVoice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint-name", default="model.pt")
    parser.add_argument("--output-dir", default="./models/static_onnx")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-feat-len", type=int, default=2048)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--no-onnxsim", action="store_true")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--encoder-only", action="store_true")
    parser.add_argument("--decoder-no-mask", action="store_true")
    parser.add_argument("--decoder-dynamic-seq", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    checkpoint_path = model_dir / args.checkpoint_name
    model_config_path = model_dir / "model.json"
    token_file = model_dir / "tokens.txt"

    for p in [checkpoint_path, model_config_path, token_file]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    tokenizer = SimpleTokenizer(str(token_file))
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config_path) as f:
        model_config = json.load(f)

    model_cls = ZipVoiceDistill if args.distill else ZipVoice
    logging.info(
        "Loading %s model from %s ...",
        "ZipVoiceDistill" if args.distill else "ZipVoice",
        model_dir,
    )
    model = model_cls(**model_config["model"], **tokenizer_config)
    model = model.to(torch.device("cpu"))
    model.eval()

    if str(checkpoint_path).endswith(".safetensors"):
        safetensors.torch.load_model(model, str(checkpoint_path))
    else:
        load_checkpoint(filename=str(checkpoint_path), model=model, strict=True)

    # CRITICAL: Replace CompactRelPositionalEncoding BEFORE convert_scaled_to_non_scaled
    # This avoids RecursiveScriptModule / If nodes in ONNX
    _replace_encoder_pos(model)

    # Convert scaled parameters for ONNX compatibility
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    # Patch SimpleDownsample only for fixed-length decoder export. The patch
    # hardwires the pad==0 trace path, which breaks dynamic-seq decoder models
    # on odd lengths like 773 because the runtime can no longer take the
    # required pad branch before downsampling.
    if not args.decoder_dynamic_seq:
        _patch_simple_downsample(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup original model state before modifications for potential retry
    import copy
    _original_model_state = copy.deepcopy(model.state_dict())

    # 1. Export encoder_core
    logging.info("=" * 60)
    logging.info("Step 1: Exporting encoder_core.onnx (max_tokens=%d)", args.max_tokens)
    encoder = EncoderCoreWrapper(model=model)
    encoder_path = output_dir / "encoder_core.onnx"
    export_encoder_core(
        model=encoder, filename=str(encoder_path),
        max_tokens=args.max_tokens, feat_dim=model.feat_dim,
        opset_version=args.opset,
    )

    encoder_onnxsim_ok = True
    if not args.no_onnxsim:
        m = onnx.load_model(str(encoder_path), load_external_data=False)
        node_count_before = len(m.graph.node)
        m = _run_onnxsim(m, "encoder_core.onnx")
        node_count_after = len(m.graph.node)
        onnx.save_model(m, str(encoder_path))
        # Check if onnxsim actually optimized (significantly reduced node count)
        if node_count_after >= node_count_before:
            logging.warning("encoder_core onnxsim did not reduce nodes (%d → %d)",
                          node_count_before, node_count_after)
            encoder_onnxsim_ok = False

    if not encoder_onnxsim_ok and not args.no_onnxsim:
        logging.warning("encoder_core onnxsim failed — trying structural fixes...")
        # Reload original model state and try re-export with additional patches
        model.load_state_dict(_original_model_state)
        model.eval()
        _replace_encoder_pos(model)
        convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)
        if not args.decoder_dynamic_seq:
            _patch_simple_downsample(model)
        # Re-export
        encoder = EncoderCoreWrapper(model=model)
        export_encoder_core(
            model=encoder, filename=str(encoder_path),
            max_tokens=args.max_tokens, feat_dim=model.feat_dim,
            opset_version=args.opset,
        )
        m = onnx.load_model(str(encoder_path), load_external_data=False)
        m = _run_onnxsim(m, "encoder_core.onnx (retry)")
        onnx.save_model(m, str(encoder_path))

    if args.encoder_only:
        runtime_config = {
            "max_tokens": args.max_tokens, "max_feat_len": args.max_feat_len,
            "feat_dim": model.feat_dim,
            "text_encoder_dim": model_config["model"]["text_encoder_dim"],
            "sampling_rate": 24000, "hop_length": 256,
            "model_type": "zipvoice_distill" if args.distill else "zipvoice",
        }
        config_path = output_dir / "runtime_config.json"
        with open(config_path, "w") as f:
            json.dump(runtime_config, f, indent=2, ensure_ascii=False)

        logging.info("=" * 60)
        logging.info("Static encoder ONNX export complete!")
        logging.info("Output: %s", output_dir)
        return

    # 2. Export fm_decoder
    logging.info("=" * 60)
    logging.info("Step 2: Exporting fm_decoder.onnx (max_feat_len=%d)", args.max_feat_len)
    decoder_cls = FmDecoderWrapperNoMask if args.decoder_no_mask else FmDecoderWrapper
    decoder = decoder_cls(model=model, distill=args.distill)
    decoder_path = output_dir / "fm_decoder.onnx"
    export_fm_decoder(
        model=decoder, filename=str(decoder_path),
        max_feat_len=args.max_feat_len,
        opset_version=args.opset,
        dynamic_seq_len=args.decoder_dynamic_seq,
    )

    if not args.no_onnxsim:
        m = onnx.load_model(str(decoder_path), load_external_data=False)
        m = _run_onnxsim(m, "fm_decoder.onnx")
        onnx.save_model(m, str(decoder_path))

    # Save runtime config
    runtime_config = {
        "max_tokens": args.max_tokens, "max_feat_len": args.max_feat_len,
        "feat_dim": model.feat_dim,
        "text_encoder_dim": model_config["model"]["text_encoder_dim"],
        "sampling_rate": 24000, "hop_length": 256,
        "model_type": "zipvoice_distill" if args.distill else "zipvoice",
    }
    config_path = output_dir / "runtime_config.json"
    with open(config_path, "w") as f:
        json.dump(runtime_config, f, indent=2, ensure_ascii=False)

    logging.info("=" * 60)
    logging.info("Static ONNX export complete!")
    logging.info("Output directory: %s", output_dir)
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        logging.info("  %-40s %8.1f MB", f.name, size_mb)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO, force=True,
    )
    main()
