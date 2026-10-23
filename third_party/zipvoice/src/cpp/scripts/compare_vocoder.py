#!/usr/bin/env python3
"""
Compare PyTorch vocoder vs quantized axmodel output.
Saves intermediate tensors for board-side comparison.

Usage:
  # Dev machine: generate test data
  python3 compare_vocoder.py --save-test-data

  # Board: run axmodel on same input, save output
  python3 compare_vocoder.py --run-axmodel

  # Dev machine: compare results
  python3 compare_vocoder.py --compare
"""
import sys, os, math, argparse
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_DIR = os.path.join(REPO_DIR, 'cpp', 'vocoder_onnx')
TEST_DIR = os.path.join(ONNX_DIR, 'test_data')
os.makedirs(TEST_DIR, exist_ok=True)

FEAT_SCALE = 0.1
N_FFT = 1024
HOP = 256


def irfft_overlap_add(real_spec, imag_spec):
    """C++ IRFFT + overlap-add, verified to match PyTorch istft (cos_sim=0.999999)."""
    n_freqs = N_FFT // 2 + 1
    T = real_spec.shape[1]
    window = 0.5 * (1.0 - np.cos(2.0 * math.pi * np.arange(N_FFT) / (N_FFT - 1)))
    window_sq = window ** 2

    # Build IRFFT basis
    irfft_cos = np.zeros((N_FFT, n_freqs), dtype=np.float32)
    irfft_sin = np.zeros((N_FFT, n_freqs), dtype=np.float32)
    for n in range(N_FFT):
        for k in range(n_freqs):
            ang = 2.0 * math.pi * k * n / N_FFT
            if k == 0:
                irfft_cos[n, k] = 1.0 / N_FFT
                irfft_sin[n, k] = 0.0
            elif k == n_freqs - 1:
                irfft_cos[n, k] = math.cos(ang) / N_FFT
                irfft_sin[n, k] = 0.0
            else:
                irfft_cos[n, k] = math.cos(ang) * (2.0 / N_FFT)
                irfft_sin[n, k] = math.sin(ang) * (2.0 / N_FFT)

    out_len = (T - 1) * HOP + N_FFT
    audio = np.zeros(out_len, dtype=np.float32)
    envelope = np.zeros(out_len, dtype=np.float32)

    for t in range(T):
        r = real_spec[0, t, :]
        im = imag_spec[0, t, :]
        frame = irfft_cos @ r - irfft_sin @ im
        pos = t * HOP
        for n in range(N_FFT):
            p = pos + n
            if p < out_len:
                audio[p] += frame[n] * window[n]
                envelope[p] += window_sq[n]

    audio /= np.maximum(envelope, 1e-10)
    pad = N_FFT // 2
    return audio[pad:pad + (T - 1) * HOP]


def compare(name, ref, test):
    ref = np.asarray(ref, dtype=np.float32).flatten()
    test = np.asarray(test, dtype=np.float32).flatten()
    if len(ref) != len(test):
        print(f"  [{name}] SIZE MISMATCH: ref={ref.shape} test={test.shape}")
        return
    diff = np.abs(ref - test)
    sig = np.mean(np.abs(ref)) + 1e-10
    cos = np.dot(ref, test) / (np.linalg.norm(ref) * np.linalg.norm(test) + 1e-10)
    print(f"  [{name}] max_err={diff.max():.2e}  rel_err={diff.mean()/sig:.2e}  cos_sim={cos:.6f}")


def cmd_save_test_data():
    """Generate test mel and save PT/ONNX reference outputs."""
    import torch
    sys.path.insert(0, REPO_DIR)
    from scripts.local_vocos import LocalVocos
    import onnxruntime as ort

    # Load models
    vocoder = LocalVocos()
    sd = torch.load(f'{REPO_DIR}/resources/vocos-mel-24khz/pytorch_model.bin',
                    weights_only=True, map_location='cpu')
    sd = {k: v for k, v in sd.items() if k.startswith(('backbone.', 'head.'))}
    vocoder.load_state_dict(sd)
    vocoder.eval()

    sess_f = ort.InferenceSession(f'{ONNX_DIR}/vocos_full_B1_T620.onnx')

    # Load real mel or generate random
    mel_bin = os.path.join(REPO_DIR, 'cpp', 'output_mel.bin')
    if os.path.exists(mel_bin):
        real_mel = np.fromfile(mel_bin, dtype=np.float32).reshape(-1, 100)  # [T, 100]
        T = real_mel.shape[0]
        # Undo feat_scale, transpose to [1, 100, T]
        mel_input = (real_mel / FEAT_SCALE).T[np.newaxis, :, :].astype(np.float32)  # [1, 100, T]
        print(f"Using real mel: shape={mel_input.shape}, range=[{mel_input.min():.3f}, {mel_input.max():.3f}]")
    else:
        T = 200
        mel_input = np.random.RandomState(42).randn(1, 100, T).astype(np.float32) * 3.0
        print(f"Using random mel: shape={mel_input.shape}")

    # Pad to 620 for ONNX
    T_pad = 620
    mel_onnx = np.zeros((1, 100, T_pad), dtype=np.float32)
    mel_onnx[:, :, :T] = mel_input[:, :, :T]

    # PT inference
    mel_pt = torch.from_numpy(mel_input)
    with torch.no_grad():
        features_pt = vocoder.backbone(mel_pt)
        audio_pt = vocoder.head(features_pt).squeeze().numpy()
        h = vocoder.head.out(features_pt)
        mag, phase = h.chunk(2, dim=-1)
        mag = torch.exp(mag).clamp(max=1e2)
        real_pt = (mag * torch.cos(phase)).numpy()
        imag_pt = (mag * torch.sin(phase)).numpy()

    # ONNX inference
    onnx_out = sess_f.run(None, {'mel': mel_onnx})
    real_onnx = onnx_out[0][:, :T, :]
    imag_onnx = onnx_out[1][:, :T, :]
    audio_onnx = irfft_overlap_add(real_onnx, imag_onnx)

    # Save everything
    np.save(f'{TEST_DIR}/mel_input.npy', mel_input)
    np.save(f'{TEST_DIR}/mel_onnx_padded.npy', mel_onnx)
    np.save(f'{TEST_DIR}/pt_real.npy', real_pt)
    np.save(f'{TEST_DIR}/pt_imag.npy', imag_pt)
    np.save(f'{TEST_DIR}/pt_audio.npy', audio_pt)
    np.save(f'{TEST_DIR}/onnx_real.npy', real_onnx)
    np.save(f'{TEST_DIR}/onnx_imag.npy', imag_onnx)
    np.save(f'{TEST_DIR}/onnx_audio.npy', audio_onnx)
    np.save(f'{TEST_DIR}/T_frames.npy', np.array([T], dtype=np.int32))

    # Meta
    with open(f'{TEST_DIR}/info.txt', 'w') as f:
        f.write(f"T={T}\n")
        f.write(f"feat_scale={FEAT_SCALE}\n")
        f.write(f"mel_range=[{mel_input.min():.4f}, {mel_input.max():.4f}]\n")
        f.write(f"pt_real_range=[{real_pt.min():.4f}, {real_pt.max():.4f}]\n")
        f.write(f"pt_audio_len={len(audio_pt)}\n")

    # Verify PT vs ONNX
    print("\n=== PT vs ONNX (dev machine) ===")
    compare('real_spectrum', real_pt, real_onnx)
    compare('imag_spectrum', imag_pt, imag_onnx)
    compare('audio', audio_pt, audio_onnx)

    # Also write audio files for listening
    import soundfile as sf
    sf.write(f'{TEST_DIR}/pt_audio.wav', audio_pt, 24000)
    sf.write(f'{TEST_DIR}/onnx_audio.wav', audio_onnx, 24000)

    print(f"\nTest data saved to {TEST_DIR}/")
    print("Copy to board and run: python3 compare_vocoder.py --run-axmodel")


def cmd_run_axmodel():
    """Run axmodel on board with the same test mel, save output."""
    import onnxruntime as ort
    from axengine import InferenceSession

    T = int(np.load(f'{TEST_DIR}/T_frames.npy')[0])
    mel_onnx = np.load(f'{TEST_DIR}/mel_onnx_padded.npy')

    # Run axmodel
    model_path = f'{ONNX_DIR}/axmodel/vocos_full.axmodel'
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        return

    print(f"Loading axmodel: {model_path}")
    sess = InferenceSession(model_path)

    print(f"Running inference (mel shape={mel_onnx.shape})...")
    outputs = sess.run(None, {'mel': mel_onnx})
    print(f"Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}")

    # Extract real/imag
    if isinstance(outputs, dict):
        real_ax = outputs['real'][:, :T, :]
        imag_ax = outputs['imag'][:, :T, :]
    elif isinstance(outputs, (list, tuple)):
        real_ax = outputs[0][:, :T, :]
        imag_ax = outputs[1][:, :T, :]
    else:
        real_ax = outputs[:, :T, :]  # guess
        imag_ax = None

    audio_ax = irfft_overlap_add(real_ax, imag_ax)

    # Save
    np.save(f'{TEST_DIR}/ax_real.npy', real_ax)
    np.save(f'{TEST_DIR}/ax_imag.npy', imag_ax)
    np.save(f'{TEST_DIR}/ax_audio.npy', audio_ax)
    import soundfile as sf
    sf.write(f'{TEST_DIR}/ax_audio.wav', audio_ax, 24000)

    # Compare with ONNX reference
    onnx_real = np.load(f'{TEST_DIR}/onnx_real.npy')
    onnx_imag = np.load(f'{TEST_DIR}/onnx_imag.npy')
    onnx_audio = np.load(f'{TEST_DIR}/onnx_audio.npy')

    print("\n=== axmodel vs ONNX (board) ===")
    compare('real_spectrum', onnx_real, real_ax)
    compare('imag_spectrum', onnx_imag, imag_ax)
    compare('audio', onnx_audio, audio_ax)

    print(f"\nResults saved to {TEST_DIR}/")
    print("Copy back to dev machine and run: python3 compare_vocoder.py --compare")


def cmd_compare():
    """Compare all outputs (dev machine, after copying ax_*.npy from board)."""
    pt_audio    = np.load(f'{TEST_DIR}/pt_audio.npy')
    onnx_audio  = np.load(f'{TEST_DIR}/onnx_audio.npy')
    ax_audio    = np.load(f'{TEST_DIR}/ax_audio.npy')

    pt_real  = np.load(f'{TEST_DIR}/pt_real.npy')
    onnx_real = np.load(f'{TEST_DIR}/onnx_real.npy')
    ax_real   = np.load(f'{TEST_DIR}/ax_real.npy')

    pt_imag  = np.load(f'{TEST_DIR}/pt_imag.npy')
    onnx_imag = np.load(f'{TEST_DIR}/onnx_imag.npy')
    ax_imag   = np.load(f'{TEST_DIR}/ax_imag.npy')

    print("=== Full Comparison ===")
    print("\n--- Spectrum ---")
    compare('real: PT vs ONNX', pt_real, onnx_real)
    compare('real: PT vs axmodel', pt_real, ax_real)
    compare('real: ONNX vs axmodel', onnx_real, ax_real)
    print()
    compare('imag: PT vs ONNX', pt_imag, onnx_imag)
    compare('imag: PT vs axmodel', pt_imag, ax_imag)
    compare('imag: ONNX vs axmodel', onnx_imag, ax_imag)

    print("\n--- Audio ---")
    compare('audio: PT vs ONNX', pt_audio, onnx_audio)
    compare('audio: PT vs axmodel', pt_audio, ax_audio)
    compare('audio: ONNX vs axmodel', onnx_audio, ax_audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-test-data', action='store_true')
    parser.add_argument('--run-axmodel', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    if args.save_test_data:
        cmd_save_test_data()
    elif args.run_axmodel:
        cmd_run_axmodel()
    elif args.compare:
        cmd_compare()
    else:
        print("Usage: --save-test-data | --run-axmodel | --compare")
        print("\nWorkflow:")
        print("  1. Dev machine:  python3 compare_vocoder.py --save-test-data")
        print("  2. Copy TEST_DIR to board, run: python3 compare_vocoder.py --run-axmodel")
        print("  3. Copy ax_*.npy back to dev, run: python3 compare_vocoder.py --compare")


if __name__ == '__main__':
    main()
