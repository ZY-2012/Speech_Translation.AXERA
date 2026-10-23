#!/usr/bin/env python3
"""Export ZipVoice static ONNX with looped attention layouts.

This keeps the original export script untouched.  The patch is process-local
and avoids the decoder attention pattern that failed AX650 check=2:

    (1024, 1, 4) --Transpose perm=[1,0,2]--> (1, 1024, 4) -> MatMul

The attention score and value paths are computed per head and per batch using
2-D matmul, then stacked back to the original ZipFormer tensor layout.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
import sys

import torch
from torch import Tensor
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import export_static_onnx  # noqa: E402

from zipvoice.models.modules import zipformer as zf  # noqa: E402


def _range_size(size) -> int:
    return int(size)


def _stack_head_batch(scores: list[list[Tensor]]) -> Tensor:
    return torch.stack([torch.stack(batch_scores, dim=0) for batch_scores in scores], dim=0)


def _relative_to_absolute(pos_scores: Tensor, seq_len: int) -> Tensor:
    rows = torch.arange(
        start=seq_len - 1,
        end=-1,
        step=-1,
        device=pos_scores.device,
        dtype=torch.long,
    ).unsqueeze(1)
    cols = torch.arange(seq_len, device=pos_scores.device, dtype=torch.long).unsqueeze(0)
    indexes = rows + cols
    return torch.gather(pos_scores, dim=1, index=indexes)


def _rel_position_forward_attn_loop(
    self: nn.Module,
    x: Tensor,
    pos_emb: Tensor,
    key_padding_mask: Tensor | None = None,
    attn_mask: Tensor | None = None,
) -> Tensor:
    x = self.in_proj(x)
    query_head_dim = self.query_head_dim
    pos_head_dim = self.pos_head_dim
    num_heads = self.num_heads

    seq_len, batch_size, _ = x.shape
    seq_len_i = _range_size(seq_len)
    batch_size_i = _range_size(batch_size)
    query_dim = query_head_dim * num_heads

    q = x[..., 0:query_dim]
    k = x[..., query_dim : 2 * query_dim]
    p = x[..., 2 * query_dim :]
    assert p.shape[-1] == num_heads * pos_head_dim, (
        p.shape[-1],
        num_heads,
        pos_head_dim,
    )

    q = self.copy_query(q)
    k = self.whiten_keys(self.balance_keys(k))
    p = self.copy_pos_query(p)

    head_scores: list[list[Tensor]] = []
    for head in range(num_heads):
        q_start = head * query_head_dim
        q_end = (head + 1) * query_head_dim
        p_start = head * pos_head_dim
        p_end = (head + 1) * pos_head_dim
        batch_scores: list[Tensor] = []
        for batch in range(batch_size_i):
            q_bh = q[:, batch, q_start:q_end]
            k_bh = k[:, batch, q_start:q_end]
            p_bh = p[:, batch, p_start:p_end]
            score = torch.matmul(q_bh, k_bh.transpose(0, 1))
            batch_scores.append(score)
        head_scores.append(batch_scores)
    attn_scores = _stack_head_batch(head_scores)

    use_pos_scores = False
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        use_pos_scores = True
    elif not self.training or random.random() >= float(self.pos_emb_skip_rate):
        use_pos_scores = True

    if use_pos_scores:
        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len_i - 1
        pos_batch_i = _range_size(pos_emb.shape[0])
        pos_head_scores: list[list[Tensor]] = []
        for head in range(num_heads):
            p_start = head * pos_head_dim
            p_end = (head + 1) * pos_head_dim
            batch_scores = []
            for batch in range(batch_size_i):
                pos_batch = 0 if pos_batch_i == 1 else batch
                p_bh = p[:, batch, p_start:p_end]
                pos_bh = pos_emb[pos_batch, :seq_len2, p_start:p_end]
                rel_scores = torch.matmul(p_bh, pos_bh.transpose(0, 1))
                batch_scores.append(_relative_to_absolute(rel_scores, seq_len_i))
            pos_head_scores.append(batch_scores)
        attn_scores = attn_scores + _stack_head_batch(pos_head_scores)

    if torch.jit.is_scripting() or torch.jit.is_tracing():
        pass
    elif self.training and random.random() < 0.1:
        attn_scores = zf.penalize_abs_values_gt(
            attn_scores, limit=25.0, penalty=1.0e-04, name=self.name
        )

    assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool
        attn_scores = attn_scores.masked_fill(attn_mask, -1000)

    if key_padding_mask is not None:
        assert key_padding_mask.shape == (batch_size, seq_len), key_padding_mask.shape
        attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1), -1000)

    attn_weights = zf.softmax(attn_scores, dim=-1)
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        pass
    elif random.random() < 0.001 and not self.training:
        self._print_attn_entropy(attn_weights)

    return nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)


def _self_attention_forward_attn_loop(
    self: nn.Module,
    x: Tensor,
    attn_weights: Tensor,
) -> Tensor:
    seq_len, batch_size, _ = x.shape
    batch_size_i = _range_size(batch_size)
    num_heads = _range_size(attn_weights.shape[0])
    assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

    x = self.in_proj(x)
    value_head_dim = x.shape[-1] // num_heads

    batch_outputs: list[Tensor] = []
    for batch in range(batch_size_i):
        head_outputs: list[Tensor] = []
        for head in range(num_heads):
            start = head * value_head_dim
            end = (head + 1) * value_head_dim
            x_bh = x[:, batch, start:end]
            weights_bh = attn_weights[head, batch]
            head_outputs.append(torch.matmul(weights_bh, x_bh))
        batch_outputs.append(torch.cat(head_outputs, dim=1).unsqueeze(1))
    x = torch.cat(batch_outputs, dim=1)

    x = self.out_proj(x)
    x = self.whiten(x)
    return x


def _nonlin_attention_forward_attn_loop(
    self: nn.Module,
    x: Tensor,
    attn_weights: Tensor,
) -> Tensor:
    x = self.in_proj(x)

    seq_len, batch_size, _ = x.shape
    batch_size_i = _range_size(batch_size)
    hidden_channels = self.hidden_channels

    s, x, y = x.chunk(3, dim=2)
    s = self.balancer(s)
    s = self.tanh(s)

    s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
    x = self.whiten1(x)
    x = x * s
    x = self.identity1(x)

    seq_len, batch_size, embed_dim = x.shape
    num_heads = _range_size(attn_weights.shape[0])
    assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

    head_dim = embed_dim // num_heads
    batch_outputs: list[Tensor] = []
    for batch in range(batch_size_i):
        head_outputs: list[Tensor] = []
        for head in range(num_heads):
            start = head * head_dim
            end = (head + 1) * head_dim
            x_bh = x[:, batch, start:end]
            weights_bh = attn_weights[head, batch]
            head_outputs.append(torch.matmul(weights_bh, x_bh))
        batch_outputs.append(torch.cat(head_outputs, dim=1).unsqueeze(1))
    x = torch.cat(batch_outputs, dim=1)

    y = self.identity2(y)
    x = x * y
    x = self.identity3(x)

    x = self.out_proj(x)
    x = self.whiten2(x)
    return x


def patch_zipformer_attention_layouts() -> None:
    zf.RelPositionMultiheadAttentionWeights.forward = _rel_position_forward_attn_loop
    zf.SelfAttention.forward = _self_attention_forward_attn_loop
    zf.NonlinAttention.forward = _nonlin_attention_forward_attn_loop
    logging.info(
        "Patched ZipFormer attention layouts with per-head/per-batch 2-D matmul loops"
    )


def main() -> None:
    patch_zipformer_attention_layouts()
    export_static_onnx.main()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    main()
