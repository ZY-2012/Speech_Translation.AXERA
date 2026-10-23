#!/usr/bin/env python3
"""
Export Vocos ONNX models with onnxsim optimization.

Static shapes (B=1, T=620):
  vocos_backbone_B1_T620.onnx     mel [1,100,620] → features [1,620,512]
  vocos_head_linear_B1_T620.onnx  features [1,620,512] → (real, imag) [1,620,513]
  vocos_full_B1_T620.onnx         mel [1,100,620] → (real, imag) [1,620,513]

IRFFT + window + overlap-add is handled in C++ (simple, ~50 lines with existing FFT).
"""
import sys, os, math
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
from scripts.local_vocos import LocalVocos, ISTFTHead

OUT_DIR = os.path.join(REPO_DIR, 'cpp', 'vocoder_onnx')
os.makedirs(OUT_DIR, exist_ok=True)


class HeadLinearOnly(nn.Module):
    """Linear(512→1026) + mag/phase → real/imag spectrum. No ISTFT."""
    def __init__(self, head: ISTFTHead):
        super().__init__()
        self.out = head.out  # Linear(512, 1026)
    def forward(self, x):    # x: [B, T, 512]
        h = self.out(x); mag, phase = h.chunk(2, dim=-1)
        mag = torch.exp(mag).clamp(max=1e2)
        return mag * torch.cos(phase), mag * torch.sin(phase)


class FullModel(nn.Module):
    def __init__(self, backbone, head_linear):
        super().__init__()
        self.backbone = backbone
        self.head_linear = head_linear
    def forward(self, mel):
        return self.head_linear(self.backbone(mel))


def export_onnx(model, dummy, path, in_names, out_names, dyn=None):
    tmp = path + '.tmp'
    torch.onnx.export(model, dummy, tmp,
                      input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=True)
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
        print(f"  [{name}/{i}] max_err={diff.max():.2e}  rel_err={diff.mean()/sig:.2e}  "
              f"shape={pt.shape}")


def main():
    print("Loading Vocos...")
    vocoder = LocalVocos()
    sd = torch.load(f'{REPO_DIR}/resources/vocos-mel-24khz/pytorch_model.bin',
                    weights_only=True, map_location='cpu')
    sd = {k: v for k, v in sd.items() if k.startswith(('backbone.', 'head.'))}
    vocoder.load_state_dict(sd)
    vocoder.eval()

    backbone = vocoder.backbone
    head_linear = HeadLinearOnly(vocoder.head).eval()

    B, T = 1, 620
    dummy_mel = torch.randn(B, 100, T)

    with torch.no_grad():
        bb_out = backbone(dummy_mel)
        pt_real, pt_imag = head_linear(bb_out)

    print(f"PT output: real={pt_real.shape}, imag={pt_imag.shape}")

    # --- Export static (B=1, T=620) ---
    print("\n=== Static models (B=1, T=620) ===")
    export_onnx(backbone, dummy_mel,
                f'{OUT_DIR}/vocos_backbone_B1_T620.onnx',
                ['mel'], ['features'])
    export_onnx(head_linear, bb_out,
                f'{OUT_DIR}/vocos_head_linear_B1_T620.onnx',
                ['features'], ['real', 'imag'])
    export_onnx(FullModel(backbone, head_linear).eval(), dummy_mel,
                f'{OUT_DIR}/vocos_full_B1_T620.onnx',
                ['mel'], ['real', 'imag'])

    # --- Verify with onnxruntime ---
    print("\n=== Verification ===")
    import onnxruntime as ort

    for tag, model_file, input_dict, pt_expected in [
        ('backbone', 'vocos_backbone_B1_T620.onnx',
         {'mel': dummy_mel.numpy()}, [bb_out.numpy()]),
        ('head_linear', 'vocos_head_linear_B1_T620.onnx',
         {'features': bb_out.numpy()}, [pt_real.numpy(), pt_imag.numpy()]),
        ('full', 'vocos_full_B1_T620.onnx',
         {'mel': dummy_mel.numpy()}, [pt_real.numpy(), pt_imag.numpy()]),
    ]:
        sess = ort.InferenceSession(f'{OUT_DIR}/{model_file}')
        onnx_out = sess.run(None, input_dict)
        verify(tag, pt_expected, onnx_out)

    print("\nDone! All exported and verified.")
    for f in sorted(os.listdir(OUT_DIR)):
        if 'B1_T620' in f:
            size = os.path.getsize(f'{OUT_DIR}/{f}') / 1024 / 1024
            print(f"  {f} ({size:.1f} MB)")


if __name__ == '__main__':
    main()
