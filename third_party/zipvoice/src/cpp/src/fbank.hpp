/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * FBANK: Mel filterbank feature extraction for prompt audio.
 *
 * Extracts log-mel spectrogram features from audio waveform, matching the
 * Python LocalVocosFbank behavior.
 **************************************************************************************************/

#pragma once

#include <vector>
#include <cmath>
#include <cstdint>

class MelFilterBank {
public:
    struct Config {
        int sampling_rate;
        int n_mels;
        int n_fft;
        int hop_length;

        Config() : sampling_rate(24000), n_mels(100), n_fft(1024), hop_length(256) {}
    };

    MelFilterBank();

    /**
     * Initialize the filterbank with given config.
     */
    int Init(const Config& config = Config());

    /**
     * Extract log-mel features from float32 audio samples [num_samples].
     * Returns features of shape [num_frames, n_mels].
     */
    std::vector<float> Extract(const std::vector<float>& samples, int sample_rate);

    /**
     * Get the number of frames for given number of samples.
     */
    static int ComputeNumFrames(int num_samples, int hop_length);

    const Config& GetConfig() const { return m_config; }

private:
    Config m_config;

    // Mel filterbank matrix [n_freqs, n_mels]
    std::vector<float> m_mel_basis;
    int m_n_freqs;

    // Hann window
    std::vector<float> m_window;

    void CreateMelFilterbank();
    void CreateWindow();

    // Simple STFT
    void ComputeSTFT(const std::vector<float>& samples,
                     std::vector<float>& spec_real,
                     std::vector<float>& spec_imag,
                     int& num_frames);
};
