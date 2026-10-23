/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Vocoder implementation using kissfft for fast IRFFT.
 **************************************************************************************************/

#include "src/vocoder.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Vocoder::Vocoder() : m_has_init(false), m_n_freqs(0) {}
Vocoder::~Vocoder() {
    m_session.reset();
    if (m_fft_cfg)  { rnn_fft_free(m_fft_cfg, 0);  m_fft_cfg = nullptr; }
    m_ifft_cfg = nullptr;  // shared with m_fft_cfg
}

int Vocoder::Init(const Config& cfg) {
    m_cfg = cfg;
    m_n_freqs = cfg.n_fft / 2 + 1;  // 513

    BuildWindow();

    // Init kissfft once
    m_fft_cfg = rnn_fft_alloc(cfg.n_fft, NULL, NULL, 0);
    if (!m_fft_cfg) {
        printf("Failed to init kissfft for vocoder\n");
        return -1;
    }
    m_ifft_cfg = m_fft_cfg;

    // Load backbone (or full) model
    m_session = std::make_unique<EngineWrapper>();
    if (m_session->Init(cfg.model_path.c_str(), 0, cfg.axclConfig) != 0) {
        printf("Failed to load vocoder model: %s\n", cfg.model_path.c_str());
        return -1;
    }

    // Load head_linear model (optional, for split mode)
    if (!cfg.head_model_path.empty()) {
        m_session_head = std::make_unique<EngineWrapper>();
        if (m_session_head->Init(cfg.head_model_path.c_str(), 0, cfg.axclConfig) != 0) {
            printf("Failed to load vocoder head model: %s\n", cfg.head_model_path.c_str());
            return -1;
        }
    }

    m_has_init = true;
    printf("Vocoder initialized: n_fft=%d, hop=%d, n_mels=%d%s\n",
           cfg.n_fft, cfg.hop_length, cfg.n_mels,
           m_session_head ? " (split mode)" : "");
    return 0;
}

void Vocoder::BuildWindow() {
    int n = m_cfg.n_fft;
    m_window.resize(n);
    m_window_sq.resize(n);
    for (int i = 0; i < n; ++i) {
        m_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (n - 1)));
        m_window_sq[i] = m_window[i] * m_window[i];
    }
}

int Vocoder::Decode(const std::vector<float>& mel, int T, float feat_scale, std::vector<float>& audio) {
    if (!m_has_init || mel.empty()) return -1;

    int feat_dim = m_cfg.n_mels;      // 100
    int n_fft = m_cfg.n_fft;          // 1024
    int n_freqs = m_n_freqs;          // 513
    int hop = m_cfg.hop_length;       // 256

    // 1. Undo feat_scale and transpose to [1, n_mels, T_model]
    float inv_scale = (feat_scale != 0.0f) ? (1.0f / feat_scale) : 1.0f;
    int T_model = 620;
    std::vector<float> mel_padded(T_model * feat_dim, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < feat_dim; ++c) {
            mel_padded[c * T_model + t] = mel[t * feat_dim + c] * inv_scale;
        }
    }

    // 2. Run NPU inference
    std::vector<float> real_spec, imag_spec;
    int spec_size;

    if (m_session_head) {
        // Split mode: backbone → features → head_linear → (real, imag)
        int in_idx = m_session->GetInputIndex("mel");
        if (in_idx < 0) in_idx = 0;
        m_session->SetInput(mel_padded.data(), in_idx);
        if (m_session->RunSync() != 0) { printf("Vocoder backbone failed\n"); return -1; }

        int feat_idx = m_session->GetOutputIndex("features");
        if (feat_idx < 0) feat_idx = 0;
        std::vector<float> features(m_session->GetOutputSize(feat_idx) / sizeof(float));
        m_session->GetOutput(features.data(), feat_idx);

        int h_in = m_session_head->GetInputIndex("features");
        if (h_in < 0) h_in = 0;
        m_session_head->SetInput(features.data(), h_in);
        if (m_session_head->RunSync() != 0) { printf("Vocoder head failed\n"); return -1; }

        int real_idx = m_session_head->GetOutputIndex("real");
        int imag_idx = m_session_head->GetOutputIndex("imag");
        if (real_idx < 0) real_idx = 0;
        if (imag_idx < 0) imag_idx = 1;
        spec_size = m_session_head->GetOutputSize(real_idx);
        real_spec.resize(spec_size / sizeof(float));
        imag_spec.resize(spec_size / sizeof(float));
        m_session_head->GetOutput(real_spec.data(), real_idx);
        m_session_head->GetOutput(imag_spec.data(), imag_idx);
    } else {
        // Full model: mel → (real, imag)
        int in_idx = m_session->GetInputIndex("mel");
        if (in_idx < 0) in_idx = 0;
        m_session->SetInput(mel_padded.data(), in_idx);
        if (m_session->RunSync() != 0) { printf("Vocoder NPU inference failed\n"); return -1; }

        int real_idx = m_session->GetOutputIndex("real");
        int imag_idx = m_session->GetOutputIndex("imag");
        if (real_idx < 0) real_idx = 0;
        if (imag_idx < 0) imag_idx = 1;
        spec_size = m_session->GetOutputSize(real_idx);
        real_spec.resize(spec_size / sizeof(float));
        imag_spec.resize(spec_size / sizeof(float));
        m_session->GetOutput(real_spec.data(), real_idx);
        m_session->GetOutput(imag_spec.data(), imag_idx);
    }

    // 3. kissfft IRFFT + window + overlap-add
    int out_len = (T - 1) * hop + n_fft;
    audio.assign(out_len, 0.0f);
    std::vector<float> envelope(out_len, 0.0f);

    std::vector<kiss_fft_cpx> cx_in(n_fft), cx_out(n_fft);

    for (int t = 0; t < T; ++t) {
        int spec_offset = t * n_freqs;

        // Build full Hermitian spectrum from [0..N/2]
        cx_in[0].r = real_spec[spec_offset + 0];
        cx_in[0].i = 0.0f;
        for (int k = 1; k < n_freqs - 1; ++k) {
            cx_in[k].r = real_spec[spec_offset + k];
            cx_in[k].i = imag_spec[spec_offset + k];
            cx_in[n_fft - k].r =  real_spec[spec_offset + k];
            cx_in[n_fft - k].i = -imag_spec[spec_offset + k];
        }
        cx_in[n_freqs - 1].r = real_spec[spec_offset + (n_freqs - 1)];
        cx_in[n_freqs - 1].i = 0.0f;  // Nyquist imag must be 0

        rnn_ifft(m_ifft_cfg, cx_in.data(), cx_out.data(), 0);

        // Window + overlap-add
        int pos = t * hop;
        for (int n = 0; n < n_fft; ++n) {
            float sample = (cx_out[n].r / n_fft) * m_window[n];
            int p = pos + n;
            if (p < out_len) {
                audio[p] += sample;
                envelope[p] += m_window_sq[n];
            }
        }
    }

    // 4. Normalize by window envelope
    for (int i = 0; i < out_len; ++i) {
        if (envelope[i] > 1e-10f) {
            audio[i] /= envelope[i];
        }
    }

    // 5. Center trim (n_fft/2 from each side, matching torch.istft center=True)
    int pad = n_fft / 2;
    int trim_len = out_len - 2 * pad;
    if (trim_len > 0) {
        std::vector<float> trimmed(audio.begin() + pad, audio.begin() + pad + trim_len);
        audio = std::move(trimmed);
    }

    return 0;
}
