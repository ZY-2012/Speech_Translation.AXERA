#!/usr/bin/env python3
"""Export full Vocos ONNX model with static shape B=1, T=620.

Only the full model is kept:

  vocos_full_B1_T620.onnx  mel [1,100,620] -> (real, imag) [1,620,513]

IRFFT + window + overlap-add is handled in C++.
"""

import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxsim import simplify


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from python.local_vocos import ISTFTHead, LocalVocos  # noqa: E402


OUT_DIR = os.path.join(ROOT_DIR, "model_convert", "vocoder_onnx")
os.makedirs(OUT_DIR, exist_ok=True)


class HeadLinearOnly(nn.Module):
    """Linear(512->1026) + mag/phase -> real/imag spectrum. No ISTFT."""

    def __init__(self, head: ISTFTHead):
        super().__init__()
        self.out = head.out

    def forward(self, x):
        h = self.out(x)
        mag, phase = h.chunk(2, dim=-1)
        mag = torch.exp(mag).clamp(max=1e2)
        return mag * torch.cos(phase), mag * torch.sin(phase)


class FullModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_linear: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head_linear = head_linear

    def forward(self, mel):
        return self.head_linear(self.backbone(mel))


def export_onnx(model, dummy, path, in_names, out_names):
    tmp = path + ".tmp"
    torch.onnx.export(
        model,
        dummy,
        tmp,
        input_names=in_names,
        output_names=out_names,
        opset_version=17,
        do_constant_folding=True,
    )
    model_simp, check = simplify(tmp)
    assert check, f"onnxsim verification failed for {path}"
    onnx.save(model_simp, path)
    os.remove(tmp)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {os.path.basename(path):50s} {size_mb:6.1f} MB")


def verify(name, pt_list, onnx_list):
    for i, (pt, onnx_val) in enumerate(zip(pt_list, onnx_list)):
        diff = np.abs(pt - onnx_val)
        sig = max(np.mean(np.abs(pt)), 1e-10)
        print(
            f"  [{name}/{i}] max_err={diff.max():.2e} "
            f"rel_err={diff.mean() / sig:.2e} shape={pt.shape}"
        )


def main():
    print("Loading Vocos...")
    vocoder = LocalVocos()
    sd = torch.load(
        f"{ROOT_DIR}/models/vocos-mel-24khz/pytorch_model.bin",
        weights_only=True,
        map_location="cpu",
    )
    sd = {k: v for k, v in sd.items() if k.startswith(("backbone.", "head."))}
    vocoder.load_state_dict(sd)
    vocoder.eval()

    backbone = vocoder.backbone
    head_linear = HeadLinearOnly(vocoder.head).eval()
    full = FullModel(backbone, head_linear).eval()

    batch, frames = 1, 620
    dummy_mel = torch.randn(batch, 100, frames)

    with torch.no_grad():
        pt_real, pt_imag = full(dummy_mel)

    print(f"PT output: real={pt_real.shape}, imag={pt_imag.shape}")

    print("\n=== Export full static model (B=1, T=620) ===")
    full_path = f"{OUT_DIR}/vocos_full_B1_T620.onnx"
    export_onnx(full, dummy_mel, full_path, ["mel"], ["real", "imag"])

    print("\n=== Verification ===")
    import onnxruntime as ort

    sess = ort.InferenceSession(full_path)
    onnx_out = sess.run(None, {"mel": dummy_mel.numpy()})
    verify("full", [pt_real.numpy(), pt_imag.numpy()], onnx_out)

    print("\nDone! Exported and verified:")
    size = os.path.getsize(full_path) / 1024 / 1024
    print(f"  vocos_full_B1_T620.onnx ({size:.1f} MB)")


if __name__ == "__main__":
    main()
