/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Vocoder: mel features → audio via vocos_full.axmodel + IRFFT + overlap-add.
 **************************************************************************************************/

#pragma once

#include <vector>
#include <string>
#include <memory>

#include "src/EngineWrapper.hpp"
#include "kiss_fft.h"

class Vocoder {
public:
    struct Config {
        std::string model_path;         // full model (or backbone in split mode)
        std::string head_model_path;    // head_linear model (optional, for split mode)
        const char* axclConfig = nullptr;
        int n_fft = 1024;
        int hop_length = 256;
        int n_mels = 100;
        int sample_rate = 24000;
    };

    Vocoder();
    ~Vocoder();

    /**
     * Initialize: load vocos_full.axmodel and pre-compute IRFFT basis.
     */
    int Init(const Config& cfg);

    /**
     * Decode mel features to audio waveform.
     * @param mel        Mel features [T, n_mels] (flat, row-major)
     * @param T          Number of mel frames
     * @param feat_scale Feature scaling factor to undo (Python: features/feat_scale before vocoder)
     * @param audio      Output audio samples (24kHz float32)
     */
    int Decode(const std::vector<float>& mel, int T, float feat_scale, std::vector<float>& audio);

    bool HasInit() const { return m_has_init; }

private:
    Config m_cfg;
    bool m_has_init;
    std::unique_ptr<EngineWrapper> m_session;
    std::unique_ptr<EngineWrapper> m_session_head;  // optional, for split mode

    // kissfft configs (shared forward/inverse twiddles)
    kiss_fft_state* m_fft_cfg = nullptr;
    kiss_fft_state* m_ifft_cfg = nullptr;

    // Hann window
    std::vector<float> m_window;

    // Window squared for envelope normalization
    std::vector<float> m_window_sq;

    int m_n_freqs;

    void BuildWindow();

    static void Fft(std::vector<float>& real, std::vector<float>& imag, bool inverse);
};
