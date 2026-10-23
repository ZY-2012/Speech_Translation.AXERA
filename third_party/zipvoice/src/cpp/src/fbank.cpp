/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * FBANK implementation with simple DFT-based STFT.
 * For production, use FFTW or an NPU-accelerated implementation.
 **************************************************************************************************/

#include "fbank.hpp"

#include <algorithm>
#include <cstring>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float hz_to_mel(float freq) {
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

MelFilterBank::MelFilterBank() : m_n_freqs(0) {}

int MelFilterBank::Init(const Config& config) {
    m_config = config;
    m_n_freqs = config.n_fft / 2 + 1;

    CreateWindow();
    CreateMelFilterbank();

    printf("FBANK init: n_mels=%d, n_fft=%d, hop=%d, sr=%d\n",
           config.n_mels, config.n_fft, config.hop_length, config.sampling_rate);
    return 0;
}

void MelFilterBank::CreateWindow() {
    m_window.resize(m_config.n_fft);
    for (int i = 0; i < m_config.n_fft; ++i) {
        m_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (m_config.n_fft - 1)));
    }
}

void MelFilterBank::CreateMelFilterbank() {
    int n_freqs = m_n_freqs;
    int n_mels = m_config.n_mels;

    // Frequency points for each FFT bin
    std::vector<float> all_freqs(n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        all_freqs[i] = i * m_config.sampling_rate / (2.0f * (n_freqs - 1));
    }

    // Mel points
    float m_min = hz_to_mel(0.0f);
    float m_max = hz_to_mel(m_config.sampling_rate / 2.0f);
    std::vector<float> m_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        m_pts[i] = m_min + (m_max - m_min) * i / (n_mels + 1);
    }

    // Convert to Hz
    std::vector<float> f_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        f_pts[i] = mel_to_hz(m_pts[i]);
    }

    // Build filterbank matrix [n_freqs, n_mels]
    m_mel_basis.resize(n_freqs * n_mels, 0.0f);

    for (int m = 0; m < n_mels; ++m) {
        for (int k = 0; k < n_freqs; ++k) {
            float freq = all_freqs[k];
            float left = f_pts[m];
            float center = f_pts[m + 1];
            float right = f_pts[m + 2];

            float val = 0.0f;
            if (freq >= left && freq <= center) {
                val = (freq - left) / (center - left);
            } else if (freq >= center && freq <= right) {
                val = (right - freq) / (right - center);
            }
            m_mel_basis[k * n_mels + m] = val;
        }
    }
}

int MelFilterBank::ComputeNumFrames(int num_samples, int hop_length) {
    return (num_samples + hop_length / 2) / hop_length;
}

// Radix-2 FFT (in-place, complex)
static void fft(std::vector<float>& real, std::vector<float>& imag, bool inverse = false) {
    int n = (int)real.size();
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    // Butterfly
    for (int len = 2; len <= n; len <<= 1) {
        float angle = 2.0f * M_PI / len * (inverse ? 1.0f : -1.0f);
        float w_real = std::cos(angle);
        float w_imag = std::sin(angle);
        for (int i = 0; i < n; i += len) {
            float cur_real = 1.0f, cur_imag = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                int a = i + j;
                int b = i + j + len / 2;
                float t_real = cur_real * real[b] - cur_imag * imag[b];
                float t_imag = cur_real * imag[b] + cur_imag * real[b];
                real[b] = real[a] - t_real;
                imag[b] = imag[a] - t_imag;
                real[a] += t_real;
                imag[a] += t_imag;
                float next_real = cur_real * w_real - cur_imag * w_imag;
                cur_imag = cur_real * w_imag + cur_imag * w_real;
                cur_real = next_real;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < n; ++i) {
            real[i] /= n;
            imag[i] /= n;
        }
    }
}

void MelFilterBank::ComputeSTFT(const std::vector<float>& samples,
                                 std::vector<float>& spec_real,
                                 std::vector<float>& spec_imag,
                                 int& num_frames) {
    int n_fft = m_config.n_fft;
    int hop = m_config.hop_length;
    int num_samples = static_cast<int>(samples.size());

    num_frames = ComputeNumFrames(num_samples, hop);
    int n_freqs = n_fft / 2 + 1;

    spec_real.assign(num_frames * n_freqs, 0.0f);
    spec_imag.assign(num_frames * n_freqs, 0.0f);

    // Pad signal (center padding, matching Python torch.stft center=True)
    int pad_amount = n_fft / 2;
    std::vector<float> padded(pad_amount + num_samples + pad_amount, 0.0f);
    std::copy(samples.begin(), samples.end(), padded.begin() + pad_amount);

    // FFT workspace
    std::vector<float> fft_real(n_fft), fft_imag(n_fft);

    for (int frame = 0; frame < num_frames; ++frame) {
        int start = frame * hop;
        // Apply window and copy to FFT buffer
        for (int n = 0; n < n_fft; ++n) {
            fft_real[n] = padded[start + n] * m_window[n];
            fft_imag[n] = 0.0f;
        }
        // Forward FFT
        fft(fft_real, fft_imag, false);
        // Extract first n_freqs bins
        for (int k = 0; k < n_freqs; ++k) {
            spec_real[frame * n_freqs + k] = fft_real[k];
            spec_imag[frame * n_freqs + k] = fft_imag[k];
        }
    }
}

std::vector<float> MelFilterBank::Extract(const std::vector<float>& samples, int sample_rate) {
    if (sample_rate != m_config.sampling_rate) {
        printf("WARNING: sample_rate mismatch: expected %d, got %d\n",
               m_config.sampling_rate, sample_rate);
    }

    int num_frames;
    std::vector<float> spec_real, spec_imag;
    ComputeSTFT(samples, spec_real, spec_imag, num_frames);

    int n_freqs = m_n_freqs;
    int n_mels = m_config.n_mels;

    // Compute magnitude spectrogram [num_frames, n_freqs]
    std::vector<float> spec_mag(num_frames * n_freqs);
    for (int i = 0; i < num_frames * n_freqs; ++i) {
        spec_mag[i] = std::sqrt(spec_real[i] * spec_real[i] + spec_imag[i] * spec_imag[i]);
    }

    // Apply mel filterbank: mel[frame, m] = sum_freq(spec[frame, freq] * basis[freq, m])
    std::vector<float> mel(num_frames * n_mels, 0.0f);
    for (int f = 0; f < num_frames; ++f) {
        for (int m = 0; m < n_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_freqs; ++k) {
                sum += spec_mag[f * n_freqs + k] * m_mel_basis[k * n_mels + m];
            }
            mel[f * n_mels + m] = std::log(std::max(sum, 1e-7f));
        }
    }

    // Expected num_frames
    int expected_frames = ComputeNumFrames(static_cast<int>(samples.size()), m_config.hop_length);
    if (num_frames > expected_frames) {
        // Truncate
        std::vector<float> truncated(expected_frames * n_mels);
        std::copy(mel.begin(), mel.begin() + expected_frames * n_mels, truncated.begin());
        return truncated;
    }

    return mel;
}
