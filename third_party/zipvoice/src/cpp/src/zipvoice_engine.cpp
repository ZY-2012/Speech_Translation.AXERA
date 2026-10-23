/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * ZipVoiceEngine implementation.
 **************************************************************************************************/

#include "zipvoice_engine.hpp"

#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <sys/time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple JSON value parser (no external dependency)
// Only parses the simple structures needed by manifest.json and runtime_config.json
namespace {

std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";

    return json.substr(pos + 1, end - pos - 1);
}

int json_get_int(const std::string& json, const std::string& key, int default_val = 0) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return default_val;

    // Skip whitespace
    pos++;
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    if (pos >= json.length()) return default_val;

    // Parse number
    std::string num_str;
    while (pos < json.length() && (std::isdigit(json[pos]) || json[pos] == '-' || json[pos] == '.')) {
        num_str += json[pos];
        pos++;
    }
    if (num_str.empty()) return default_val;

    // Check if float
    if (num_str.find('.') != std::string::npos) {
        return static_cast<int>(std::stof(num_str));
    }
    return std::stoi(num_str);
}

float json_get_float(const std::string& json, const std::string& key, float default_val = 0.0f) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return default_val;

    pos++;
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    if (pos >= json.length()) return default_val;

    std::string num_str;
    while (pos < json.length() && (std::isdigit(json[pos]) || json[pos] == '-' || json[pos] == '.')) {
        num_str += json[pos];
        pos++;
    }
    if (num_str.empty()) return default_val;
    return std::stof(num_str);
}

std::vector<std::string> json_get_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;

    pos = json.find('[', pos + search.length());
    if (pos == std::string::npos) return result;

    size_t end = json.find(']', pos);
    if (end == std::string::npos) return result;

    std::string array_str = json.substr(pos + 1, end - pos - 1);

    size_t start = 0;
    while (start < array_str.length()) {
        size_t q1 = array_str.find('"', start);
        if (q1 == std::string::npos) break;
        size_t q2 = array_str.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        result.push_back(array_str.substr(q1 + 1, q2 - q1 - 1));
        start = q2 + 1;
    }
    return result;
}

std::string read_file_content(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // anonymous namespace

ZipVoiceEngine::ZipVoiceEngine()
    : m_has_init(false), m_decoder_seq_len(0), m_decoder_has_padding_mask(false) {}

ZipVoiceEngine::~ZipVoiceEngine() {
    m_sessions.clear();
}

double ZipVoiceEngine::GetCurrentTimeMs() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int ZipVoiceEngine::Init(const std::string& model_dir, const char* axclConfig) {
    if (LoadConfig(model_dir) != 0) return -1;
    if (LoadManifest(model_dir) != 0) return -1;
    if (LoadModels(model_dir, axclConfig) != 0) return -1;
    if (LoadDecoderMetadata() != 0) return -1;

    m_has_init = true;
    printf("ZipVoiceEngine initialized: max_tokens=%d, max_feat_len=%d, feat_dim=%d, num_step=%d\n",
           m_config.max_tokens, m_config.max_feat_len, m_config.feat_dim, m_config.num_step);
    return 0;
}

int ZipVoiceEngine::LoadConfig(const std::string& model_dir) {
    std::string config_path = model_dir + "/runtime_config.json";
    std::string json = read_file_content(config_path);
    if (!json.empty()) {
        m_config.max_tokens = json_get_int(json, "max_tokens", 384);
        m_config.max_feat_len = json_get_int(json, "max_feat_len", 1024);
        m_config.feat_dim = json_get_int(json, "feat_dim", 100);
        m_config.sampling_rate = json_get_int(json, "sampling_rate", 24000);
        m_config.hop_length = json_get_int(json, "hop_length", 256);
        m_config.num_step = json_get_int(json, "num_step", 10);
        m_config.t_shift = json_get_float(json, "t_shift", 0.5f);
        m_config.guidance_scale = json_get_float(json, "guidance_scale", 1.0f);
    }
    m_config.model_dir = model_dir;
    return 0;
}

int ZipVoiceEngine::LoadManifest(const std::string& model_dir) {
    std::string manifest_path = model_dir + "/decoder4_split_manifest.json";
    std::string json = read_file_content(manifest_path);
    if (json.empty()) {
        printf("Failed to read manifest: %s\n", manifest_path.c_str());
        return -1;
    }

    // Parse encoder info
    m_encoder_info.name = json_get_string(json, "name");
    // Need to parse the nested "encoder" object
    size_t enc_pos = json.find("\"encoder\"");
    if (enc_pos != std::string::npos) {
        size_t obj_start = json.find('{', enc_pos);
        size_t obj_end = json.find('}', obj_start);
        if (obj_start != std::string::npos && obj_end != std::string::npos) {
            std::string enc_json = json.substr(obj_start, obj_end - obj_start + 1);
            m_encoder_info.name = json_get_string(enc_json, "name");
            m_encoder_info.file = json_get_string(enc_json, "file");
            m_encoder_info.inputs = json_get_string_array(enc_json, "inputs");
            m_encoder_info.outputs = json_get_string_array(enc_json, "outputs");
        }
    }

    // Parse decoder_parts array
    size_t dec_pos = json.find("\"decoder_parts\"");
    if (dec_pos != std::string::npos) {
        size_t arr_start = json.find('[', dec_pos);
        if (arr_start != std::string::npos) {
            m_decoder_parts.clear();
            size_t pos = arr_start + 1;
            int depth = 0;
            std::string part_json;
            for (; pos < json.length(); ++pos) {
                if (json[pos] == '{') depth++;
                if (depth > 0) part_json += json[pos];
                if (json[pos] == '}') {
                    depth--;
                    if (depth == 0) {
                        // Parse this part
                        ModelInfo info;
                        info.name = json_get_string(part_json, "name");
                        info.file = json_get_string(part_json, "file");
                        info.inputs = json_get_string_array(part_json, "inputs");
                        info.outputs = json_get_string_array(part_json, "outputs");
                        m_decoder_parts.push_back(info);
                        part_json.clear();
                    }
                }
            }
        }
    }

    printf("Manifest: encoder=%s, decoder_parts=%zu\n",
           m_encoder_info.name.c_str(), m_decoder_parts.size());
    for (size_t i = 0; i < m_decoder_parts.size(); ++i) {
        printf("  part%zu: %s (%s) in=%zu out=%zu\n", i,
               m_decoder_parts[i].name.c_str(), m_decoder_parts[i].file.c_str(),
               m_decoder_parts[i].inputs.size(), m_decoder_parts[i].outputs.size());
    }
    return 0;
}

int ZipVoiceEngine::LoadModels(const std::string& model_dir, const char* axclConfig) {
    // Load encoder
    std::string enc_path = model_dir + "/" + m_encoder_info.file;
    auto enc = std::make_unique<EngineWrapper>();
    if (enc->Init(enc_path.c_str(), 0, axclConfig) != 0) {
        printf("Failed to load encoder: %s\n", enc_path.c_str());
        return -1;
    }
    m_sessions[m_encoder_info.name] = std::move(enc);

    // Load decoder parts
    for (auto& part : m_decoder_parts) {
        std::string path = model_dir + "/" + part.file;
        auto sess = std::make_unique<EngineWrapper>();
        if (sess->Init(path.c_str(), 0, axclConfig) != 0) {
            printf("Failed to load decoder part: %s\n", path.c_str());
            return -1;
        }
        m_sessions[part.name] = std::move(sess);
    }

    printf("Loaded %zu models\n", m_sessions.size());
    return 0;
}

int ZipVoiceEngine::LoadDecoderMetadata() {
    if (m_decoder_parts.empty()) return -1;

    auto& part0 = m_decoder_parts[0];
    auto it = m_sessions.find(part0.name);
    if (it == m_sessions.end()) return -1;

    auto& sess = it->second;

    // Check if padding_mask is an input
    int pad_idx = sess->GetInputIndex("padding_mask");
    m_decoder_has_padding_mask = (pad_idx >= 0);

    // Get sequence length from x input shape
    m_decoder_seq_len = m_config.max_feat_len;

    // Try to get seq_len from model metadata
    // The shape info is in the AX engine io_info; for simplicity we use config value

    printf("Decoder metadata: seq_len=%d, has_padding_mask=%d\n",
           m_decoder_seq_len, m_decoder_has_padding_mask ? 1 : 0);
    return 0;
}

std::vector<float> ZipVoiceEngine::GetTimesteps(int num_step, float t_shift) const {
    std::vector<float> ts(num_step + 1);
    for (int i = 0; i <= num_step; ++i) {
        float t = static_cast<float>(i) / num_step;
        ts[i] = t_shift * t / (1.0f + (t_shift - 1.0f) * t);
    }
    return ts;
}

int ZipVoiceEngine::RunEncoder(const std::vector<int32_t>& cat_tokens,
                                std::vector<float>& out_encoded) {
    auto it = m_sessions.find(m_encoder_info.name);
    if (it == m_sessions.end()) return -1;

    auto& sess = it->second;

    // Set input
    int input_idx = sess->GetInputIndex(m_encoder_info.inputs[0].c_str());
    if (input_idx < 0) input_idx = 0;

    // Need to copy to non-const buffer for the engine API
    std::vector<int32_t> input_copy = cat_tokens;
    sess->SetInput(input_copy.data(), input_idx);

    // Run
    if (sess->RunSync() != 0) return -1;

    // Get output
    int output_idx = sess->GetOutputIndex(m_encoder_info.outputs[0].c_str());
    if (output_idx < 0) output_idx = 0;

    int output_size = sess->GetOutputSize(output_idx);
    out_encoded.resize(output_size / sizeof(float));
    sess->GetOutput(out_encoded.data(), output_idx);

    return 0;
}

int ZipVoiceEngine::DurationExpand(const std::vector<float>& encoded,
                                    int prompt_tokens_len,
                                    int text_tokens_len,
                                    int prompt_features_len,
                                    float speed,
                                    std::vector<float>& out_text_condition,
                                    int& out_features_len) {
    int total_tokens_len = prompt_tokens_len + text_tokens_len;
    if (total_tokens_len <= 0) return -1;

    // Compute target features length
    int features_len = static_cast<int>(
        std::ceil(static_cast<float>(prompt_features_len) / prompt_tokens_len
                  * total_tokens_len / speed)
    );
    if (features_len > m_config.max_feat_len) {
        features_len = m_config.max_feat_len;
    }

    int feat_dim = m_config.feat_dim;

    // encoded shape: [1, max_tokens, feat_dim] (flat array)
    // Extract the no-pad portion
    int token_dur = features_len / total_tokens_len;
    int max_tokens = m_config.max_tokens;

    // out_text_condition: [1, features_len, feat_dim]
    out_text_condition.assign(features_len * feat_dim, 0.0f);

    // Repeat each token embedding token_dur times
    for (int t = 0; t < total_tokens_len; ++t) {
        int base_idx = t * feat_dim;
        for (int d = 0; d < token_dur; ++d) {
            int frame_idx = t * token_dur + d;
            if (frame_idx >= features_len) break;
            int dst_idx = frame_idx * feat_dim;
            std::copy(encoded.begin() + base_idx,
                      encoded.begin() + base_idx + feat_dim,
                      out_text_condition.begin() + dst_idx);
        }
    }

    // Fill residual frames with last token embedding
    int filled = total_tokens_len * token_dur;
    int residual = features_len - filled;
    if (residual > 0) {
        int last_base = total_tokens_len * feat_dim;
        for (int d = 0; d < residual; ++d) {
            int dst_idx = (filled + d) * feat_dim;
            std::copy(encoded.begin() + last_base,
                      encoded.begin() + last_base + feat_dim,
                      out_text_condition.begin() + dst_idx);
        }
    }

    out_features_len = features_len;
    return 0;
}

int ZipVoiceEngine::RunDecoderPart(const ModelInfo& part,
                                    std::map<std::string, std::vector<float>>& values,
                                    const std::vector<uint8_t>* padding_mask_data,
                                    std::vector<uint8_t>* padding_mask2_out) {
    auto it = m_sessions.find(part.name);
    if (it == m_sessions.end()) return -1;
    auto& sess = it->second;

    // Set inputs in order
    for (size_t i = 0; i < part.inputs.size(); ++i) {
        const std::string& expected_name = part.inputs[i];

        int input_idx = sess->GetInputIndex(expected_name.c_str());
        if (input_idx < 0) input_idx = static_cast<int>(i);

        // padding_mask: use uint8 data from caller (part0 only)
        if (expected_name == "padding_mask") {
            if (padding_mask_data) {
                sess->SetInput((void*)padding_mask_data->data(), input_idx);
                continue;
            }
        }
        // padding_mask2: use raw uint8 data from part0's output
        if (expected_name == "padding_mask2") {
            if (padding_mask2_out && !padding_mask2_out->empty()) {
                sess->SetInput((void*)padding_mask2_out->data(), input_idx);
                continue;
            }
            printf("ERROR: padding_mask2 not available for '%s'\n", part.name.c_str());
            return -1;
        }

        // Find the value in float map
        auto val_it = values.find(expected_name);
        if (val_it == values.end()) {
            printf("Missing input '%s' for model '%s'\n",
                   expected_name.c_str(), part.name.c_str());
            return -1;
        }

        sess->SetInput((void*)val_it->second.data(), input_idx);
    }

    // Run
    if (sess->RunSync() != 0) return -1;

    // Get outputs
    for (size_t i = 0; i < part.outputs.size(); ++i) {
        const std::string& name = part.outputs[i];

        int output_idx = sess->GetOutputIndex(name.c_str());
        if (output_idx < 0) output_idx = static_cast<int>(i);

        int size = sess->GetOutputSize(output_idx);
        if (size <= 0) {
            printf("Invalid output size for '%s' in '%s'\n",
                   name.c_str(), part.name.c_str());
            return -1;
        }

        // padding_mask2 is uint8 — read raw bytes, don't convert to float
        if (name == "padding_mask2" && padding_mask2_out) {
            padding_mask2_out->resize(size);
            sess->GetOutput(padding_mask2_out->data(), output_idx);
            continue;
        }

        std::vector<float> output_data(size / sizeof(float));
        sess->GetOutput(output_data.data(), output_idx);
        values[name] = std::move(output_data);
    }

    return 0;
}

int ZipVoiceEngine::Sample(const std::vector<int32_t>& cat_tokens,
                            int prompt_tokens_len,
                            int text_tokens_len,
                            const std::vector<float>& prompt_features,
                            int prompt_features_len,
                            float speed,
                            float guidance_scale,
                            int seed,
                            std::vector<float>& out_features,
                            Timing& out_timing) {
    double t_total_start = GetCurrentTimeMs();

    // 1. Run encoder
    double t_start = GetCurrentTimeMs();
    std::vector<float> encoded;
    if (RunEncoder(cat_tokens, encoded) != 0) return -1;
    out_timing.encoder_time_sec = static_cast<float>(GetCurrentTimeMs() - t_start) / 1000.0f;

    // encoded shape: [max_tokens * feat_dim] (flattened, batch=1)
    int feat_dim = m_config.feat_dim;

    // 2. Duration expand
    t_start = GetCurrentTimeMs();
    std::vector<float> text_condition;
    int features_len;
    if (DurationExpand(encoded, prompt_tokens_len, text_tokens_len,
                       prompt_features_len, speed,
                       text_condition, features_len) != 0) return -1;
    out_timing.duration_expand_time_sec = static_cast<float>(GetCurrentTimeMs() - t_start) / 1000.0f;

    // 3. Prepare decoder inputs
    int seq_len = m_decoder_seq_len > 0 ? m_decoder_seq_len : m_config.max_feat_len;

    if (features_len > seq_len) {
        printf("features_len=%d exceeds decoder seq_len=%d\n", features_len, seq_len);
        return -1;
    }
    if (prompt_features_len > seq_len) {
        printf("prompt_features_len=%d exceeds decoder seq_len=%d\n", prompt_features_len, seq_len);
        return -1;
    }

    // text_cond_padded: [1, seq_len, feat_dim] (zeros, filled up to features_len)
    std::vector<float> text_cond_padded(seq_len * feat_dim, 0.0f);
    std::copy(text_condition.begin(),
              text_condition.begin() + features_len * feat_dim,
              text_cond_padded.begin());

    // speech_cond_padded: [1, seq_len, feat_dim]
    std::vector<float> speech_cond_padded(seq_len * feat_dim, 0.0f);
    std::copy(prompt_features.begin(),
              prompt_features.begin() + prompt_features_len * feat_dim,
              speech_cond_padded.begin());
    // DEBUG

    // padding_mask: [1, seq_len] as uint8_t (bool), NOT float32!
    // Python: np.zeros((1, seq_len), dtype=np.bool_)
    std::vector<uint8_t> padding_mask(seq_len, 0);
    for (int i = features_len; i < seq_len; ++i) {
        padding_mask[i] = 1;
    }

    // x: random init [1, seq_len, feat_dim]
    std::vector<float> x(seq_len * feat_dim, 0.0f);
    // Simple LCG random
    uint32_t rng_state = static_cast<uint32_t>(seed > 0 ? seed : 42);
    for (int i = 0; i < features_len * feat_dim; ++i) {
        rng_state = rng_state * 1103515245 + 12345;
        // Box-Muller for normal distribution
        float u1 = static_cast<float>(rng_state & 0x7FFFFFFF) / 0x7FFFFFFF;
        rng_state = rng_state * 1103515245 + 12345;
        float u2 = static_cast<float>(rng_state & 0x7FFFFFFF) / 0x7FFFFFFF;
        x[i] = std::sqrt(-2.0f * std::log(std::max(u1, 1e-10f)))
               * std::cos(2.0f * M_PI * u2);
    }

    // Time steps
    std::vector<float> timesteps = GetTimesteps(m_config.num_step, m_config.t_shift);

    // Guidance scale as single float array
    std::vector<float> gs = { guidance_scale };

    // 4. Flow-matching decoder
    double t_dec_total = 0.0;
    for (int step = 0; step < m_config.num_step; ++step) {
        double t_step_start = GetCurrentTimeMs();

        // Prepare inputs for this step
        std::map<std::string, std::vector<float>> values;

        // t: scalar
        std::vector<float> t_val = { timesteps[step] };
        values["t"] = t_val;

        // x: [1, seq_len, feat_dim]
        values["x"] = x;

        // text_condition
        values["text_condition"] = text_cond_padded;

        // speech_condition
        values["speech_condition"] = speech_cond_padded;

        // guidance_scale: scalar
        values["guidance_scale"] = gs;

        // padding_mask is passed separately as uint8_t (bool), not in values map
        std::vector<uint8_t> padding_mask2_data;  // populated by part0, used by parts 1-3

        // Run all decoder parts in sequence
        for (auto& part : m_decoder_parts) {
            if (RunDecoderPart(part, values, &padding_mask, &padding_mask2_data) != 0) return -1;
        }

        t_dec_total += GetCurrentTimeMs() - t_step_start;

        // Get v (velocity) from last part output
        const std::string& final_output_name = m_decoder_parts.back().outputs[0];
        auto vit = values.find(final_output_name);
        if (vit == values.end()) {
            printf("Missing final output '%s'\n", final_output_name.c_str());
            return -1;
        }

        const std::vector<float>& v = vit->second;

        // Euler step: x = x + v * dt
        float dt = timesteps[step + 1] - timesteps[step];
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += v[i] * dt;
        }

        // Zero out padding region
        for (int i = features_len * feat_dim; i < seq_len * feat_dim; ++i) {
            x[i] = 0.0f;
        }
    }

    out_timing.decoder_time_sec = static_cast<float>(t_dec_total) / 1000.0f;

    // 5. Extract generated features (excluding prompt region)
    int generated_frames = features_len - prompt_features_len;
    if (generated_frames <= 0) {
        generated_frames = features_len;
        out_features.assign(x.begin(), x.begin() + features_len * feat_dim);
    } else {
        int offset = prompt_features_len * feat_dim;
        out_features.assign(x.begin() + offset,
                            x.begin() + features_len * feat_dim);
    }

    out_timing.generated_frames = generated_frames;
    out_timing.features_len = features_len;
    out_timing.total_time_sec = static_cast<float>(GetCurrentTimeMs() - t_total_start) / 1000.0f;

    printf("  encoder: %.3f s  dur_expand: %.3f s  decoder(%d steps): %.3f s (avg %.3f ms/step)  total: %.3f s\n",
           out_timing.encoder_time_sec,
           out_timing.duration_expand_time_sec,
           m_config.num_step,
           out_timing.decoder_time_sec,
           out_timing.decoder_time_sec / m_config.num_step * 1000.0f,
           out_timing.total_time_sec);

    return 0;
}
