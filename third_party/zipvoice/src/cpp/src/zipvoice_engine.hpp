/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * ZipVoiceEngine: the core TTS inference engine.
 *
 * Manages encoder + split-decoder4 axmodel inference, duration expansion,
 * and flow-matching sampling. Mirrors the Python Decoder4ZipVoiceBoardRuntime.
 *
 * Model pipeline:
 *   1. encoder.axmodel:  cat_tokens[n] → encoded[n, feat_dim]
 *   2. Duration expand:  repeat tokens → text_condition[1, seq_len, feat_dim]
 *   3. decoder_part0..3: flow-matching ODE solver (Euler, num_step steps)
 *      part0: t, x, text_cond, speech_cond, guidance_scale, padding_mask
 *             → decoder_hidden_p0, time_hidden, padding_mask2, cfg_scale
 *      part1: decoder_hidden_p0, time_hidden, padding_mask2 → decoder_hidden_p1
 *      part2: decoder_hidden_p1, time_hidden, padding_mask2 → decoder_hidden_p2
 *      part3: decoder_hidden_p2, time_hidden, padding_mask2, cfg_scale → v
 *      x = x + v * dt
 **************************************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

#include "EngineWrapper.hpp"

class ZipVoiceEngine {
public:
    struct Config {
        std::string model_dir;
        int max_feat_len = 1024;
        int max_tokens = 384;
        int feat_dim = 100;
        int sampling_rate = 24000;
        int hop_length = 256;
        int num_step = 10;
        float t_shift = 0.5f;
        float guidance_scale = 1.0f;
    };

    struct ModelInfo {
        std::string name;
        std::string file;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
    };

    struct Timing {
        float encoder_time_sec = 0.0f;
        float duration_expand_time_sec = 0.0f;
        float decoder_time_sec = 0.0f;
        float total_time_sec = 0.0f;
        int generated_frames = 0;
        int features_len = 0;
    };

    ZipVoiceEngine();
    ~ZipVoiceEngine();

    /**
     * Initialize from a model directory.
     * @param model_dir   Path containing axmodel files and JSON configs
     * @param axclConfig  AXCL only: path to axcl.json (nullptr → default). Ignored on AXERA.
     */
    int Init(const std::string& model_dir, const char* axclConfig = nullptr);

    /**
     * Run encoder: cat_tokens[1, max_tokens] → encoded[1, max_tokens, feat_dim].
     * out_encoded should be pre-allocated with size: 1 * max_tokens * feat_dim floats.
     */
    int RunEncoder(const std::vector<int32_t>& cat_tokens, std::vector<float>& out_encoded);

    /**
     * Duration expand: token-level features → frame-level features.
     */
    int DurationExpand(const std::vector<float>& encoded,
                       int prompt_tokens_len,
                       int text_tokens_len,
                       int prompt_features_len,
                       float speed,
                       std::vector<float>& out_text_condition,
                       int& out_features_len);

    /**
     * Full sampling: encoder → duration_expand → decoder flow-matching.
     *
     * @param cat_tokens          [max_tokens] int32 token IDs
     * @param prompt_tokens_len   number of prompt tokens
     * @param text_tokens_len     number of text tokens
     * @param prompt_features     [prompt_frames, feat_dim] mel features from prompt wav
     * @param prompt_features_len number of prompt feature frames
     * @param speed               speed factor (1.0 = normal)
     * @param guidance_scale      classifier-free guidance scale
     * @param seed                random seed
     * @param out_features        output mel features [generated_frames, feat_dim]
     * @param out_timing          timing breakdown
     */
    int Sample(const std::vector<int32_t>& cat_tokens,
               int prompt_tokens_len,
               int text_tokens_len,
               const std::vector<float>& prompt_features,
               int prompt_features_len,
               float speed,
               float guidance_scale,
               int seed,
               std::vector<float>& out_features,
               Timing& out_timing);

    const Config& GetConfig() const { return m_config; }
    void SetNumStep(int n) { if (n >= 1) m_config.num_step = n; }
    bool HasInit() const { return m_has_init; }

private:
    Config m_config;
    bool m_has_init;

    ModelInfo m_encoder_info;
    std::vector<ModelInfo> m_decoder_parts;

    // Model sessions (one per axmodel)
    std::map<std::string, std::unique_ptr<EngineWrapper>> m_sessions;

    // Decoder metadata
    int m_decoder_seq_len;
    bool m_decoder_has_padding_mask;

    int LoadConfig(const std::string& model_dir);
    int LoadManifest(const std::string& model_dir);
    int LoadModels(const std::string& model_dir, const char* axclConfig = nullptr);
    int LoadDecoderMetadata();

    std::vector<float> GetTimesteps(int num_step, float t_shift) const;

    int RunDecoderPart(const ModelInfo& part,
                       std::map<std::string, std::vector<float>>& values,
                       const std::vector<uint8_t>* padding_mask_data = nullptr,
                       std::vector<uint8_t>* padding_mask2_out = nullptr);

    static double GetCurrentTimeMs();
};
