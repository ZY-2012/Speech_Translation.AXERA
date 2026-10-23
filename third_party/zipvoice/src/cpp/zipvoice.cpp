/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Main entry: Command-line TTS inference using ZipVoice encoder + split-decoder4
 * axmodels on AXERA NPU boards.
 *
 * Pipeline:
 *   1. Load tokenizer (tokens.txt)
 *   2. Load prompt audio → extract mel filterbank features
 *   3. Tokenize text → build cat_tokens
 *   4. Encoder (encoder.axmodel) → encoded features
 *   5. Duration expand
 *   6. Decoder4 (decoder_part0..3.axmodel) → flow-matching → mel features
 *   7. Save mel features as raw float32 binary (can be decoded with Python vocoder)
 *
 * Usage:
 *   ./zipvoice_axera \
 *     --model-dir ../models/zipvoice_ax650 \
 *     --token-file ../resources/zipvoice_hf/zipvoice/tokens.txt \
 *     --prompt-wav ../assets/moss_prompts/zh_1_4p5s.wav \
 *     --prompt-text "你好，欢迎使用语音合成系统" \
 *     --text "这是要合成的目标文本" \
 *     --output-feat output_mel.bin
 *
 * Requirements: AX650 board with axengine SDK
 *
 * Based on ZipVoice.AXERA Python inference and melotts.axera-main C++ patterns.
 **************************************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <unistd.h>
#include <sys/wait.h>

#if defined(AX650) || defined(AX630C) || defined(AX620Q)
#include "ax_sys_api.h"
#endif

#include "src/cmdline.hpp"
#include "src/EngineWrapper.hpp"
#include "src/tokenizer.hpp"
#include "src/fbank.hpp"
#include "src/zipvoice_engine.hpp"
#include "src/vocoder.hpp"
#include "src/wav_writer.hpp"

static double get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Persistent Python daemon for fast tokenization (imports jieba/pypinyin once at startup).
// Communication via pipe: C++ writes commands to daemon's stdin, reads responses from stdout.
struct PyDaemon {
    int pid = -1;
    FILE* write_pipe = nullptr;
    FILE* read_pipe = nullptr;

    ~PyDaemon() { Stop(); }

    bool Start(const std::string& repo_dir, const std::string& daemon_script) {
        int pipe_to_cpp[2], pipe_to_py[2];
        if (pipe(pipe_to_cpp) != 0 || pipe(pipe_to_py) != 0) return false;

        pid = fork();
        if (pid < 0) { close(pipe_to_cpp[0]); close(pipe_to_cpp[1]); close(pipe_to_py[0]); close(pipe_to_py[1]); return false; }

        if (pid == 0) {
            // Child: Python daemon
            close(pipe_to_cpp[0]);  // close read end of cpp-pipe
            close(pipe_to_py[1]);   // close write end of py-pipe
            dup2(pipe_to_py[0], STDIN_FILENO);
            dup2(pipe_to_cpp[1], STDOUT_FILENO);
            close(pipe_to_cpp[1]); close(pipe_to_py[0]);
            execlp("python3", "python3", "-u", daemon_script.c_str(), repo_dir.c_str(), nullptr);
            _exit(1);
        }

        // Parent
        close(pipe_to_cpp[1]);  // close write end
        close(pipe_to_py[0]);   // close read end
        write_pipe = fdopen(pipe_to_py[1], "w");
        read_pipe  = fdopen(pipe_to_cpp[0], "r");
        if (!write_pipe || !read_pipe) { Stop(); return false; }
        setlinebuf(write_pipe);

        // Wait for READY signal
        char buf[256];
        if (!fgets(buf, sizeof(buf), read_pipe) || strncmp(buf, "READY", 5) != 0) {
            printf("PyDaemon: failed to start (%s)\n", buf ? buf : "no response");
            Stop(); return false;
        }
        printf("PyDaemon: started\n");
        return true;
    }

    void Stop() {
        if (write_pipe) { fprintf(write_pipe, "quit\n"); fflush(write_pipe); fclose(write_pipe); write_pipe = nullptr; }
        if (read_pipe)  { fclose(read_pipe); read_pipe = nullptr; }
        if (pid > 0)    { waitpid(pid, nullptr, 0); pid = -1; }
    }
};

static int DaemonCountTokens(PyDaemon& py_daemon, const std::string& prompt_file,
                             const std::string& text_file, int* prompt_len, int* text_len) {
    if (!py_daemon.write_pipe || !py_daemon.read_pipe) return -1;
    fprintf(py_daemon.write_pipe, "count\t%s\t%s\n", prompt_file.c_str(), text_file.c_str());
    fflush(py_daemon.write_pipe);
    char buf[256];
    if (fgets(buf, sizeof(buf), py_daemon.read_pipe) && sscanf(buf, "COUNT %d %d", prompt_len, text_len) == 2) {
        return 0;
    }
    return -1;
}

static int DaemonTokenize(PyDaemon& py_daemon, const std::string& prompt_file,
                          const std::string& text_file, int max_tokens,
                          const std::string& output_bin, int* prompt_len, int* text_len) {
    if (!py_daemon.write_pipe || !py_daemon.read_pipe) return -1;
    fprintf(py_daemon.write_pipe, "tokenize\t%s\t%s\t%d\t%s\n",
            prompt_file.c_str(), text_file.c_str(), max_tokens, output_bin.c_str());
    fflush(py_daemon.write_pipe);
    char buf[256];
    if (fgets(buf, sizeof(buf), py_daemon.read_pipe)) {
        if (sscanf(buf, "TOKENS %d %d", prompt_len, text_len) == 2) return 0;
        if (strncmp(buf, "ERROR", 5) == 0) {
            printf("PyDaemon: %s", buf);
            return -2;
        }
    }
    return -1;
}

/**
 * Read float32 WAV file and return samples.
 * Supports 16-bit PCM and 32-bit float WAV files with basic header parsing.
 */
static int ReadWavFile(const std::string& path, std::vector<float>& samples, int& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        printf("Failed to open: %s\n", path.c_str());
        return -1;
    }

    // Read RIFF header
    char riff[5] = {};
    file.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        printf("Not a valid WAV file: %s\n", path.c_str());
        return -1;
    }

    uint32_t file_size;
    file.read(reinterpret_cast<char*>(&file_size), 4);

    char wave[5] = {};
    file.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        printf("Not a valid WAV file: %s\n", path.c_str());
        return -1;
    }

    // Parse chunks
    int num_channels = 1;
    int bits_per_sample = 16;
    sample_rate = 24000;
    uint32_t data_size = 0;

    while (file.good()) {
        char chunk_id[5] = {};
        file.read(chunk_id, 4);
        uint32_t chunk_size;
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format, num_ch, bps;
            uint32_t sr, byte_rate;
            uint16_t block_align;

            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_ch), 2);
            file.read(reinterpret_cast<char*>(&sr), 4);
            file.read(reinterpret_cast<char*>(&byte_rate), 4);
            file.read(reinterpret_cast<char*>(&block_align), 2);
            file.read(reinterpret_cast<char*>(&bps), 2);

            num_channels = num_ch;
            sample_rate = sr;
            bits_per_sample = bps;

            // Skip remaining fmt bytes
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            break;
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (data_size == 0) {
        printf("No data chunk found in WAV: %s\n", path.c_str());
        return -1;
    }

    // Read audio data
    int num_samples = data_size / (bits_per_sample / 8) / num_channels;

    if (bits_per_sample == 16) {
        std::vector<int16_t> raw(num_samples * num_channels);
        file.read(reinterpret_cast<char*>(raw.data()), data_size);

        samples.resize(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            // Take first channel only
            samples[i] = raw[i * num_channels] / 32768.0f;
        }
    } else if (bits_per_sample == 32) {
        // Assume float32
        samples.resize(num_samples * num_channels);
        file.read(reinterpret_cast<char*>(samples.data()), data_size);

        // Extract first channel
        std::vector<float> mono(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            mono[i] = samples[i * num_channels];
        }
        samples = std::move(mono);
    } else {
        printf("Unsupported bit depth: %d\n", bits_per_sample);
        return -1;
    }

    printf("Read WAV: %s, sr=%d, ch=%d, samples=%d\n",
           path.c_str(), sample_rate, num_channels, num_samples);
    return 0;
}

/**
 * Simple linear resampling.
 */
static std::vector<float> ResampleLinear(const std::vector<float>& samples,
                                          int orig_sr, int target_sr) {
    if (orig_sr == target_sr) return samples;

    int old_len = static_cast<int>(samples.size());
    int new_len = std::max(1, static_cast<int>(
        std::round(static_cast<double>(old_len) * target_sr / orig_sr)));

    std::vector<float> result(new_len);
    for (int i = 0; i < new_len; ++i) {
        double pos = static_cast<double>(i) * (old_len - 1) / (new_len - 1);
        int idx = static_cast<int>(pos);
        double frac = pos - idx;

        if (idx + 1 < old_len) {
            result[i] = static_cast<float>(
                samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else {
            result[i] = samples[old_len - 1];
        }
    }
    return result;
}

/**
 * Compute RMS of audio samples.
 */
static float ComputeRms(const std::vector<float>& samples) {
    if (samples.empty()) return 0.0f;
    float sum_sq = 0.0f;
    for (float s : samples) sum_sq += s * s;
    return std::sqrt(sum_sq / samples.size());
}

/**
 * RMS normalize audio.
 */
static void RmsNormalize(std::vector<float>& samples, float target_rms) {
    float rms = ComputeRms(samples);
    if (rms < target_rms && rms > 1e-10f) {
        float gain = target_rms / rms;
        for (float& s : samples) s *= gain;
    }
}

static bool IsUtf8Lead(unsigned char c) {
    return (c & 0xC0) != 0x80;
}

static std::vector<std::string> SplitUtf8Chars(const std::string& text) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = (unsigned char)text[i];
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

static bool IsChineseUtf8Char(const std::string& ch) {
    if (ch.size() != 3) return false;
    unsigned char b0 = (unsigned char)ch[0];
    unsigned char b1 = (unsigned char)ch[1];
    unsigned char b2 = (unsigned char)ch[2];
    uint32_t cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

static bool IsSplitPunct(const std::string& ch) {
    static const char* puncts[] = {
        ".", "!", "?", ";", ",", ":",
        "。", "！", "？", "；", "，", "、", "："
    };
    for (auto* p : puncts) if (ch == p) return true;
    return false;
}

static std::string TrimAsciiSpaces(const std::string& s) {
    size_t start = 0, end = s.size();
    while (start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) start++;
    while (end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r')) end--;
    return s.substr(start, end - start);
}

static std::string JoinUnits(const std::string& left, const std::string& right) {
    if (left.empty()) return TrimAsciiSpaces(right);
    std::string r = TrimAsciiSpaces(right);
    if (r.empty()) return TrimAsciiSpaces(left);
    auto left_chars = SplitUtf8Chars(left);
    auto right_chars = SplitUtf8Chars(r);
    bool zh_boundary = (!left_chars.empty() && IsChineseUtf8Char(left_chars.back())) ||
                       (!right_chars.empty() && IsChineseUtf8Char(right_chars.front()));
    return zh_boundary ? (TrimAsciiSpaces(left) + r) : (TrimAsciiSpaces(left) + " " + r);
}

static std::vector<std::string> SplitUnitsCpp(const std::string& text) {
    std::vector<std::string> units;
    auto chars = SplitUtf8Chars(TrimAsciiSpaces(text));
    std::string current;
    for (const auto& ch : chars) {
        current += ch;
        if (IsSplitPunct(ch)) {
            std::string t = TrimAsciiSpaces(current);
            if (!t.empty()) units.push_back(t);
            current.clear();
        }
    }
    current = TrimAsciiSpaces(current);
    if (!current.empty()) units.push_back(current);
    if (units.empty() && !text.empty()) units.push_back(TrimAsciiSpaces(text));
    return units;
}

static int TokenCountCpp(Tokenizer& tokenizer, const std::string& text) {
    return (int)tokenizer.TextToTokenIds(text).size();
}

struct SegmentInfoCpp {
    std::string text;
    int text_tokens = 0;
    int raw_features_len = 0;
    int features_len = 0;
    int generated_frames = 0;
};

static std::vector<std::string> SplitLongUnitCpp(Tokenizer& tokenizer, const std::string& unit, int max_text_tokens) {
    if (TokenCountCpp(tokenizer, unit) <= max_text_tokens) return {unit};

    std::vector<std::string> chunks;
    if (unit.find(' ') != std::string::npos) {
        std::stringstream ss(unit);
        std::string piece, current;
        while (ss >> piece) {
            std::string candidate = JoinUnits(current, piece);
            if (!current.empty() && TokenCountCpp(tokenizer, candidate) > max_text_tokens) {
                chunks.push_back(current);
                current = piece;
            } else {
                current = candidate;
            }
        }
        if (!current.empty()) chunks.push_back(current);
        return chunks;
    }

    auto chars = SplitUtf8Chars(unit);
    std::string current;
    for (const auto& ch : chars) {
        std::string candidate = current + ch;
        if (!current.empty() && TokenCountCpp(tokenizer, candidate) > max_text_tokens) {
            chunks.push_back(current);
            current = ch;
        } else {
            current = candidate;
        }
    }
    if (!current.empty()) chunks.push_back(current);
    return chunks;
}

static SegmentInfoCpp EstimateSegmentCpp(Tokenizer& tokenizer, const std::string& text,
                                         int prompt_frames, int prompt_tokens_len,
                                         float speed, int max_feat_len) {
    SegmentInfoCpp s;
    s.text = text;
    s.text_tokens = TokenCountCpp(tokenizer, text);
    s.raw_features_len = (int)std::ceil((double)prompt_frames / prompt_tokens_len * (prompt_tokens_len + s.text_tokens) / speed);
    s.features_len = std::min(s.raw_features_len, max_feat_len);
    s.generated_frames = s.features_len - prompt_frames;
    if (s.generated_frames <= 0) s.generated_frames = s.features_len;
    return s;
}

static std::vector<SegmentInfoCpp> BuildSegmentsCpp(Tokenizer& tokenizer, const std::string& text,
                                                    int prompt_frames, int prompt_tokens_len,
                                                    float speed, int max_feat_len,
                                                    int max_text_tokens, int min_generated_frames,
                                                    int max_generated_frames, double max_raw_feat_ratio) {
    auto raw_units = SplitUnitsCpp(text);
    std::vector<std::string> units;
    for (const auto& u : raw_units) {
        auto split = SplitLongUnitCpp(tokenizer, u, max_text_tokens);
        units.insert(units.end(), split.begin(), split.end());
    }

    std::vector<SegmentInfoCpp> segments;
    std::string current;
    for (const auto& unit : units) {
        std::string candidate = JoinUnits(current, unit);
        auto cand = EstimateSegmentCpp(tokenizer, candidate, prompt_frames, prompt_tokens_len, speed, max_feat_len);
        bool raw_too_long = cand.raw_features_len > (int)(max_feat_len * max_raw_feat_ratio);
        bool too_long = cand.text_tokens > max_text_tokens || cand.generated_frames > max_generated_frames || raw_too_long;
        if (!current.empty() && too_long) {
            segments.push_back(EstimateSegmentCpp(tokenizer, current, prompt_frames, prompt_tokens_len, speed, max_feat_len));
            current = unit;
        } else {
            current = candidate;
        }
    }
    if (!current.empty()) segments.push_back(EstimateSegmentCpp(tokenizer, current, prompt_frames, prompt_tokens_len, speed, max_feat_len));

    if (segments.size() >= 2 && segments.back().generated_frames < min_generated_frames) {
        std::string merged_text = JoinUnits(segments[segments.size()-2].text, segments.back().text);
        auto merged = EstimateSegmentCpp(tokenizer, merged_text, prompt_frames, prompt_tokens_len, speed, max_feat_len);
        bool raw_ok = merged.raw_features_len <= (int)(max_feat_len * max_raw_feat_ratio);
        if (merged.text_tokens <= max_text_tokens && merged.generated_frames <= max_generated_frames && raw_ok) {
            segments[segments.size()-2] = merged;
            segments.pop_back();
        }
    }
    return segments;
}

static std::vector<std::string> SplitLongUnitCppDaemon(PyDaemon& py_daemon,
                                                        const std::string& prompt_text,
                                                        const std::string& unit,
                                                        int max_text_tokens) {
    std::string tmp_prompt = "/tmp/zipvoice_prompt_count.txt";
    std::string tmp_text = "/tmp/zipvoice_text_count.txt";
    {
        std::ofstream pf(tmp_prompt); pf << prompt_text; pf.close();
    }

    auto count_for = [&](const std::string& txt) -> int {
        std::ofstream tf(tmp_text); tf << txt; tf.close();
        int p = 0, t = 0;
        if (DaemonCountTokens(py_daemon, tmp_prompt, tmp_text, &p, &t) != 0) return 1000000;
        return t;
    };

    if (count_for(unit) <= max_text_tokens) return {unit};

    std::vector<std::string> chunks;
    if (unit.find(' ') != std::string::npos) {
        std::stringstream ss(unit);
        std::string piece, current;
        while (ss >> piece) {
            std::string candidate = JoinUnits(current, piece);
            if (!current.empty() && count_for(candidate) > max_text_tokens) {
                chunks.push_back(current);
                current = piece;
            } else {
                current = candidate;
            }
        }
        if (!current.empty()) chunks.push_back(current);
        return chunks;
    }

    auto chars = SplitUtf8Chars(unit);
    std::string current;
    for (const auto& ch : chars) {
        std::string candidate = current + ch;
        if (!current.empty() && count_for(candidate) > max_text_tokens) {
            chunks.push_back(current);
            current = ch;
        } else {
            current = candidate;
        }
    }
    if (!current.empty()) chunks.push_back(current);
    return chunks;
}

static std::vector<SegmentInfoCpp> BuildSegmentsCppDaemon(PyDaemon& py_daemon,
                                                          const std::string& prompt_text,
                                                          const std::string& text,
                                                          int prompt_frames,
                                                          int prompt_tokens_len,
                                                          float speed,
                                                          int max_feat_len,
                                                          int max_text_tokens,
                                                          int min_generated_frames,
                                                          int max_generated_frames,
                                                          double max_raw_feat_ratio) {
    std::string tmp_prompt = "/tmp/zipvoice_prompt_count.txt";
    std::string tmp_text = "/tmp/zipvoice_text_count.txt";
    {
        std::ofstream pf(tmp_prompt); pf << prompt_text; pf.close();
    }

    auto token_count_daemon = [&](const std::string& txt) -> int {
        std::ofstream tf(tmp_text); tf << txt; tf.close();
        int p = 0, t = 0;
        if (DaemonCountTokens(py_daemon, tmp_prompt, tmp_text, &p, &t) != 0) return 1000000;
        return t;
    };
    auto estimate_segment = [&](const std::string& seg_text) -> SegmentInfoCpp {
        SegmentInfoCpp s;
        s.text = seg_text;
        s.text_tokens = token_count_daemon(seg_text);
        s.raw_features_len = (int)std::ceil((double)prompt_frames / prompt_tokens_len * (prompt_tokens_len + s.text_tokens) / speed);
        s.features_len = std::min(s.raw_features_len, max_feat_len);
        s.generated_frames = s.features_len - prompt_frames;
        if (s.generated_frames <= 0) s.generated_frames = s.features_len;
        return s;
    };

    auto raw_units = SplitUnitsCpp(text);
    std::vector<std::string> units;
    for (const auto& u : raw_units) {
        auto split = SplitLongUnitCppDaemon(py_daemon, prompt_text, u, max_text_tokens);
        units.insert(units.end(), split.begin(), split.end());
    }

    std::vector<SegmentInfoCpp> segments;
    std::string current;
    for (const auto& unit : units) {
        std::string candidate = JoinUnits(current, unit);
        auto cand = estimate_segment(candidate);
        bool raw_too_long = cand.raw_features_len > (int)(max_feat_len * max_raw_feat_ratio);
        bool too_long = cand.text_tokens > max_text_tokens || cand.generated_frames > max_generated_frames || raw_too_long;
        if (!current.empty() && too_long) {
            segments.push_back(estimate_segment(current));
            current = unit;
        } else {
            current = candidate;
        }
    }
    if (!current.empty()) segments.push_back(estimate_segment(current));

    if (segments.size() >= 2 && segments.back().generated_frames < min_generated_frames) {
        std::string merged_text = JoinUnits(segments[segments.size()-2].text, segments.back().text);
        auto merged = estimate_segment(merged_text);
        bool raw_ok = merged.raw_features_len <= (int)(max_feat_len * max_raw_feat_ratio);
        if (merged.text_tokens <= max_text_tokens && merged.generated_frames <= max_generated_frames && raw_ok) {
            segments[segments.size()-2] = merged;
            segments.pop_back();
        }
    }
    return segments;
}


/**
 * Save features as raw float32 binary (compatible with numpy .fromfile).
 */
static bool SaveFeaturesBin(const std::string& path,
                             const std::vector<float>& features,
                             int num_frames, int feat_dim) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.write(reinterpret_cast<const char*>(features.data()),
               num_frames * feat_dim * sizeof(float));
    file.close();
    printf("Saved features: %s [%d, %d]\n", path.c_str(), num_frames, feat_dim);
    return true;
}

int main(int argc, char** argv) {
    // --- Command line parsing ---
    cmdline::parser cmd;

    cmd.add<std::string>("model-dir", 'm', "Model directory containing axmodels and configs",
                         true, "");
    cmd.add<std::string>("token-file", 't', "Path to tokens.txt",
                         false, "");
    cmd.add<std::string>("prompt-wav", 'w', "Prompt audio WAV file", true, "");
    cmd.add<std::string>("prompt-text", 'p', "Prompt text (for tokenization; optional with --cat-tokens-file)", false, "");
    cmd.add<std::string>("text", 's', "Text to synthesize", false, "");
    cmd.add<std::string>("text-file", 'f', "UTF-8 text file to synthesize", false, "");
    cmd.add<std::string>("output-wav", 'o', "Output WAV file path",
                         false, "output.wav");
    cmd.add<std::string>("output-feat", 0, "Output mel features as raw float32 binary (optional)",
                         false, "");
    cmd.add<std::string>("repo-dir", 0, "Repo root dir. When set, auto-decode mel to WAV via Python vocoder (deprecated, use --vocoder-model).",
                         false, "");
    cmd.add<std::string>("vocoder-model", 0, "Path to vocos_full.axmodel for C++ vocoder. When set, all-C++ pipeline, no Python.",
                         false, "");
    cmd.add<std::string>("vocoder-head-model", 0, "AX630C: path to head_linear axmodel (split mode, optional)",
                         false, "");
    cmd.add<std::string>("axcl-config", 0, "AXCL only: path to axcl.json config file",
                         false, "/usr/local/axcl/axcl.json");
    cmd.add<int>("device-index", 0, "AXCL only: device index (0=first card)", false, 0);
    cmd.add<std::string>("cat-tokens-file", 0, "Pre-computed cat_tokens int32 binary (from export_tokens.py). "
                         "When set, skips built-in tokenizer.",
                         false, "");
    cmd.add<int>("prompt-tokens-len", 0, "Number of prompt tokens (required with --cat-tokens-file)", false, 0);
    cmd.add<int>("text-tokens-len", 0, "Number of text tokens (required with --cat-tokens-file)", false, 0);

    cmd.add<int>("max-tokens", 0, "Max token sequence length", false, 384);
    cmd.add<int>("max-feat-len", 0, "Max feature sequence length", false, 1024);
    cmd.add<int>("num-step", 0, "Number of flow-matching steps", false, 10);
    cmd.add<float>("speed", 0, "Speech speed factor", false, 1.0f);
    cmd.add<float>("guidance-scale", 0, "CFG guidance scale (0=use config)", false, 0.0f);
    cmd.add<float>("t-shift", 0, "Time shift for flow scheduler", false, 0.5f);
    cmd.add<float>("feat-scale", 0, "Feature scaling factor", false, 0.1f);
    cmd.add<float>("target-rms", 0, "Target RMS for audio normalization", false, 0.1f);
    cmd.add<int>("seed", 0, "Random seed", false, 42);
    cmd.add<int>("min-generated-frames", 0, "Minimum generated frames per segment",
                 false, 360);
    cmd.add<int>("max-generated-frames", 0, "Maximum generated frames per segment",
                 false, 620);

    cmd.parse_check(argc, argv);

    auto model_dir    = cmd.get<std::string>("model-dir");
    auto token_file   = cmd.get<std::string>("token-file");
    auto prompt_wav   = cmd.get<std::string>("prompt-wav");
    auto prompt_text  = cmd.get<std::string>("prompt-text");
    auto text         = cmd.get<std::string>("text");
    auto text_file    = cmd.get<std::string>("text-file");
    auto output_wav   = cmd.get<std::string>("output-wav");
    auto output_feat  = cmd.get<std::string>("output-feat");
    auto repo_dir     = cmd.get<std::string>("repo-dir");
    auto vocoder_model = cmd.get<std::string>("vocoder-model");
    auto vocoder_head_model = cmd.get<std::string>("vocoder-head-model");
    auto axcl_config   = cmd.get<std::string>("axcl-config");
    int  device_index  = cmd.get<int>("device-index");
    auto cat_tokens_file = cmd.get<std::string>("cat-tokens-file");
    int prompt_tokens_len_cmd = cmd.get<int>("prompt-tokens-len");
    int text_tokens_len_cmd   = cmd.get<int>("text-tokens-len");

    int max_tokens      = cmd.get<int>("max-tokens");
    int max_feat_len    = cmd.get<int>("max-feat-len");
    int num_step        = cmd.get<int>("num-step");
    float speed         = cmd.get<float>("speed");
    float guidance_scale = cmd.get<float>("guidance-scale");
    float t_shift       = cmd.get<float>("t-shift");
    float feat_scale    = cmd.get<float>("feat-scale");
    float target_rms    = cmd.get<float>("target-rms");
    int seed            = cmd.get<int>("seed");

    // Load target text (not required when using pre-computed tokens)
    std::string target_text = "(from cat-tokens-file)";
    if (!cat_tokens_file.empty()) {
        // Text not needed; pre-computed tokens are used
    } else if (!text.empty()) {
        target_text = text;
    } else if (!text_file.empty()) {
        std::ifstream file(text_file);
        if (!file.is_open()) {
            printf("Failed to open text file: %s\n", text_file.c_str());
            return -1;
        }
        std::stringstream ss;
        ss << file.rdbuf();
        target_text = ss.str();
    } else {
        printf("ERROR: Either --text, --text-file, or --cat-tokens-file is required\n");
        return -1;
    }

    // --- Load tokenizer first (needed for Python-aligned segmentation) ---
    double t_tokenizer = get_current_time_ms();
    Tokenizer tokenizer;
    if (!token_file.empty()) {
        if (tokenizer.Load(token_file) != 0) {
            printf("Failed to load tokenizer\n");
            return -1;
        }
        printf("Tokenizer load: %.0f ms\n", get_current_time_ms() - t_tokenizer);
    } else {
        printf("WARNING: No token file provided. Token IDs must be pre-computed.\n");
    }

    // Normalize whitespace only (Python load_text behavior)
    {
        std::string normalized;
        bool last_was_space = false;
        for (char c : target_text) {
            if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            if (c == ' ') { if (!last_was_space) normalized += c; last_was_space = true; }
            else { normalized += c; last_was_space = false; }
        }
        while (!normalized.empty() && normalized.back() == ' ') normalized.pop_back();
        target_text = normalized;
    }

    // --- Start Python daemon early (needed for English long-text segmentation) ---
    PyDaemon py_daemon;
    if (!repo_dir.empty()) {
        std::string daemon_path = repo_dir + "/cpp/scripts/py_daemon.py";
        py_daemon.Start(repo_dir, daemon_path);
    }

    // Python-aligned long-text segmentation (text_processing.build_segments)
    std::vector<std::string> sentences;
    {
        int prompt_tokens_len_actual = 0;
        if (!repo_dir.empty()) {
            // English: get true prompt token length from daemon
            std::string tmp_prompt = "/tmp/zipvoice_prompt_seg.txt";
            std::string tmp_dummy = "/tmp/zipvoice_dummy_seg.txt";
            { std::ofstream pf(tmp_prompt); pf << prompt_text; pf.close(); }
            { std::ofstream df(tmp_dummy); df << "x"; df.close(); }
            int p = 0, t = 0;
            if (DaemonCountTokens(py_daemon, tmp_prompt, tmp_dummy, &p, &t) != 0) {
                printf("ERROR: failed to count prompt tokens via daemon\n");
                return -1;
            }
            prompt_tokens_len_actual = p;
        } else {
            // Chinese: C++ tokenizer
            prompt_tokens_len_actual = TokenCountCpp(tokenizer, prompt_text);
        }

        int max_text_tokens = max_tokens - prompt_tokens_len_actual - 1;
        int min_generated_frames = 360;
        int max_generated_frames = 620;
        double max_raw_feat_ratio = 1.2;
        if (!repo_dir.empty()) {
            // English path: use Python daemon token counts inside build_segments
            auto segments = BuildSegmentsCppDaemon(py_daemon, prompt_text, target_text,
                                                   422, prompt_tokens_len_actual,
                                                   speed, max_feat_len,
                                                   max_text_tokens,
                                                   min_generated_frames,
                                                   max_generated_frames,
                                                   max_raw_feat_ratio);
            for (const auto& seg : segments) sentences.push_back(seg.text);
        } else {
            // Chinese path: use C++ tokenizer token counts
            auto segments = BuildSegmentsCpp(tokenizer, target_text,
                                             422, prompt_tokens_len_actual,
                                             speed, max_feat_len,
                                             max_text_tokens,
                                             min_generated_frames,
                                             max_generated_frames,
                                             max_raw_feat_ratio);
            for (const auto& seg : segments) sentences.push_back(seg.text);
        }
    }

    printf("After build_segments: %zu segments\n", sentences.size());

    printf("========================================\n");
    printf("ZipVoice AXERA C++ Inference\n");
    printf("========================================\n");
    printf("model-dir:     %s\n", model_dir.c_str());
    printf("token-file:    %s\n", token_file.c_str());
    printf("prompt-wav:    %s\n", prompt_wav.c_str());
    printf("prompt-text:   %s\n", prompt_text.c_str());
    printf("target-text:   %s\n", target_text.c_str());
    printf("output-wav:    %s\n", output_wav.c_str());
    printf("max-tokens:    %d\n", max_tokens);
    printf("max-feat-len:  %d\n", max_feat_len);
    printf("num-step:      %d\n", num_step);
    printf("speed:         %.2f\n", speed);
    printf("guidance-scale:%.2f\n", guidance_scale);
    printf("t-shift:       %.2f\n", t_shift);
    printf("seed:          %d\n", seed);
    printf("========================================\n");

    // --- Init AX system (AXERA only; AXCL handles init internally) ---
#if defined(AX650) || defined(AX630C) || defined(AX620Q)
    double t_total_start = get_current_time_ms();
    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "AX_ENGINE_Init failed{0x%8x}.\n", ret);
        return -1;
    }
#endif

    // tokenizer already loaded above for Python-aligned segmentation

    // --- Load prompt audio & extract features ---
    double t_feat_start = get_current_time_ms();
    std::vector<float> prompt_audio;
    int prompt_sr;
    if (ReadWavFile(prompt_wav, prompt_audio, prompt_sr) != 0) {
        return -1;
    }

    // Resample to 24kHz if needed
    std::vector<float> prompt_resampled = ResampleLinear(prompt_audio, prompt_sr, 24000);

    // RMS normalize
    float prompt_rms = ComputeRms(prompt_resampled);
    RmsNormalize(prompt_resampled, target_rms);

    // Extract mel filterbank features
    MelFilterBank fbank;
    MelFilterBank::Config fbank_cfg;
    fbank_cfg.sampling_rate = 24000;
    fbank_cfg.n_mels = 100;
    fbank_cfg.n_fft = 1024;
    fbank_cfg.hop_length = 256;
    fbank.Init(fbank_cfg);

    std::vector<float> prompt_mel = fbank.Extract(prompt_resampled, 24000);
    int prompt_frames = MelFilterBank::ComputeNumFrames(
        static_cast<int>(prompt_resampled.size()), 256);

    // Scale features
    for (float& v : prompt_mel) v *= feat_scale;

    printf("Prompt features: %d frames x %d mels (%.0f ms)\n",
           prompt_frames, fbank_cfg.n_mels,
           get_current_time_ms() - t_feat_start);
    double t_fbank_ms = get_current_time_ms() - t_feat_start;

    // ---- RTF timing starts here (excludes one-time init) ----
    double t_rtf_start = get_current_time_ms();

    // --- Initialize ZipVoice engine (once, before sentence loop) ---
    double t_load_start = get_current_time_ms();
    ZipVoiceEngine engine;

    if (engine.Init(model_dir, axcl_config.c_str()) != 0) {
        printf("Failed to initialize ZipVoice engine\n");
        return -1;
    }
    printf("Engine load: %.0f ms\n", get_current_time_ms() - t_load_start);

    // --- Load vocoder axmodel (once, before sentence loop) ---
    Vocoder vocoder;
    if (!vocoder_model.empty()) {
        double t_vocoder_load = get_current_time_ms();
        Vocoder::Config vcfg;
        vcfg.model_path = vocoder_model;
        vcfg.head_model_path = vocoder_head_model;
        vcfg.axclConfig = axcl_config.c_str();
        if (vocoder.Init(vcfg) != 0) {
            printf("ERROR: Failed to load vocoder model\n");
            return -1;
        }
        printf("Vocoder load: %.0f ms\n", get_current_time_ms() - t_vocoder_load);
    }

    // --- Process sentences ---
    std::vector<float> all_audio;
    double total_npu_ms = 0.0;
    double total_vocoder_ms = 0.0;
    int silence_samples = (int)(24000 * 0.14);  // 140ms silence between sentences (matching Python)
    std::vector<float> silence(silence_samples, 0.0f);

    for (size_t si = 0; si < sentences.size(); ++si) {
        std::string target_text = sentences[si];
        if (sentences.size() > 1) printf("\n--- Sentence %zu/%zu: %s ---\n", si+1, sentences.size(), target_text.substr(0, 60).c_str());

    double t_tokenize = get_current_time_ms();

    std::vector<int> prompt_tokens;
    std::vector<int> text_tokens;
    int prompt_tokens_len = 0;
    int text_tokens_len = 0;

    std::string actual_cat_tokens_file = cat_tokens_file;
    std::string tmp_tokens_file;

    if (!cat_tokens_file.empty()) {
        // Use pre-computed cat_tokens file
        prompt_tokens_len = prompt_tokens_len_cmd;
        text_tokens_len   = text_tokens_len_cmd;
        if (prompt_tokens_len <= 0 || text_tokens_len <= 0) {
            printf("ERROR: --prompt-tokens-len and --text-tokens-len required with --cat-tokens-file\n");
            return -1;
        }
        printf("Using pre-computed tokens: %s (prompt=%d, text=%d)\n",
               cat_tokens_file.c_str(), prompt_tokens_len, text_tokens_len);
    } else if (tokenizer.IsLoaded()) {
        // Fast: use built-in C++ tokenizer (requires --token-file)
        prompt_tokens = tokenizer.TextToTokenIds(prompt_text);
        text_tokens = tokenizer.TextToTokenIds(target_text);
        prompt_tokens_len = (int)prompt_tokens.size();
        text_tokens_len = (int)text_tokens.size();
        printf("C++ tokenizer: prompt=%d, text=%d (%.0f ms)\n",
               prompt_tokens_len, text_tokens_len, get_current_time_ms() - t_tokenize);
    } else if (!repo_dir.empty()) {
        // Fast: use persistent Python daemon (imports done once at startup)
        std::string tmp_prompt = "/tmp/zipvoice_prompt.txt";
        std::string tmp_target = "/tmp/zipvoice_target.txt";
        tmp_tokens_file = "/tmp/zipvoice_cat_tokens.bin";
        {
            std::ofstream pf(tmp_prompt); pf << prompt_text; pf.close();
            std::ofstream tf(tmp_target); tf << target_text; tf.close();
        }
        if (DaemonTokenize(py_daemon, tmp_prompt, tmp_target, max_tokens,
                           tmp_tokens_file, &prompt_tokens_len, &text_tokens_len) == 0) {
            actual_cat_tokens_file = tmp_tokens_file;
            printf("Python tokenizer: prompt=%d, text=%d (%.0f ms)\n",
                   prompt_tokens_len, text_tokens_len, get_current_time_ms() - t_tokenize);
        } else {
            printf("ERROR: Python daemon tokenization failed\n");
            return -1;
        }
    } else {
        printf("ERROR: Provide --token-file, --cat-tokens-file, or --repo-dir\n");
        return -1;
    }

    // Build or load cat_tokens [prompt + text + pad]
    std::vector<int32_t> cat_tokens;
    if (!actual_cat_tokens_file.empty()) {
        // Read int32 binary
        std::ifstream ctf(actual_cat_tokens_file, std::ios::binary);
        if (!ctf.is_open()) {
            printf("Failed to open: %s\n", actual_cat_tokens_file.c_str());
            return -1;
        }
        ctf.seekg(0, std::ios::end);
        size_t file_size = ctf.tellg();
        ctf.seekg(0, std::ios::beg);
        cat_tokens.resize(file_size / sizeof(int32_t));
        ctf.read(reinterpret_cast<char*>(cat_tokens.data()), file_size);
        ctf.close();
        printf("Loaded %zu cat_tokens\n", cat_tokens.size());
        if ((int)cat_tokens.size() > max_tokens) {
            max_tokens = (int)cat_tokens.size();
        }
    } else if (tokenizer.IsLoaded()) {
        tokenizer.BuildCatTokens(prompt_tokens, text_tokens, max_tokens, cat_tokens);
    } else {
        printf("ERROR: Cannot build cat_tokens\n");
        return -1;
    }

    int feat_dim = fbank_cfg.n_mels;

    ZipVoiceEngine::Timing timing;
    std::vector<float> output_features;

    if (guidance_scale == 0.0f) guidance_scale = engine.GetConfig().guidance_scale;
    if (engine.Sample(cat_tokens,
                      prompt_tokens_len,
                      text_tokens_len,
                      prompt_mel,
                      prompt_frames,
                      speed,
                      guidance_scale,
                      seed,
                      output_features,
                      timing) != 0) {
        printf("Inference failed!\n");
        return -1;
    }

    double t_infer_end = get_current_time_ms();

    int generated_frames = timing.generated_frames;
    total_npu_ms += timing.total_time_sec * 1000.0;

    // --- Save mel features ---
    if (!output_feat.empty()) {
        SaveFeaturesBin(output_feat, output_features, generated_frames, feat_dim);
    }

    // --- Decode mel to WAV ---
    float audio_sec = 0.0f;
    if (!vocoder_model.empty()) {
        // C++ vocoder (axmodel): fast, all NPU
        printf("\nDecoding via C++ vocoder...\n");
        double t_vocoder_start = get_current_time_ms();

        std::vector<float> audio;
        if (vocoder.Decode(output_features, generated_frames, feat_scale, audio) != 0) {
            printf("ERROR: Vocoder decode failed\n");
            return -1;
        }

        double t_vocoder_decode = get_current_time_ms() - t_vocoder_start;
        audio_sec = audio.size() / 24000.0f;

        // RMS normalize (matching Python vocoder_decode_loaded)
        RmsNormalize(audio, target_rms);
        if (prompt_rms < target_rms) {
            float scale = prompt_rms / target_rms;
            for (float& s : audio) s *= scale;
        }

        // Accumulate audio + silence between sentences
        if (!all_audio.empty()) all_audio.insert(all_audio.end(), silence.begin(), silence.end());
        all_audio.insert(all_audio.end(), audio.begin(), audio.end());
        total_vocoder_ms += t_vocoder_decode;

        double t_rtf_ms = get_current_time_ms() - t_rtf_start;
        double t_npu_ms = timing.total_time_sec * 1000.0;

    } else if (!repo_dir.empty()) {
        // Python vocoder (deprecated fallback)
        printf("\nDecoding mel to WAV via Python vocoder...\n");
        double t_vocoder_start = get_current_time_ms();
        char cmdline[2048];
        snprintf(cmdline, sizeof(cmdline),
            "python3 -c \""
            "import numpy as np, soundfile as sf, sys;"
            "sys.path.insert(0, '%s');"
            "from scripts.common_infer import load_vocoder, vocoder_decode_loaded;"
            "feat = np.fromfile('%s', dtype=np.float32).reshape(1, -1, 100);"
            "v = load_vocoder('%s');"
            "audio = vocoder_decode_loaded(v, feat, feat_scale=%.2f, target_rms=%.2f, prompt_rms=%.4f);"
            "sf.write('%s', audio, 24000);"
            "print('AUDIO_SEC=%%f' %% (len(audio)/24000))"
            "\"",
            repo_dir.c_str(), output_feat.c_str(), repo_dir.c_str(),
            feat_scale, target_rms, prompt_rms,
            output_wav.c_str());
        FILE* fp = popen(cmdline, "r");
        if (fp) {
            char buf[256];
            while (fgets(buf, sizeof(buf), fp)) {
                printf("%s", buf);
                sscanf(buf, "AUDIO_SEC=%f", &audio_sec);
            }
            pclose(fp);
        }
        double t_vocoder = get_current_time_ms() - t_vocoder_start;
        double t_rtf_ms = get_current_time_ms() - t_rtf_start;
        double t_npu_ms = timing.total_time_sec * 1000.0;
        printf("  NPU inference: %.0f ms  vocoder: %.0f ms\n", t_npu_ms, t_vocoder);
        if (audio_sec > 0.0f) {
            printf("  RTF (NPU only): %.4f (%.3f s / audio %.2f s)\n",
                   t_npu_ms / 1000.0 / audio_sec, t_npu_ms / 1000.0, audio_sec);
        }
        total_vocoder_ms += t_vocoder;
    } else {
        printf("\nTo decode mel to WAV, add --vocoder-model <path/to/vocos_full.axmodel>\n");
    }
    }  // end for each sentence

    // Write concatenated audio
    if (!all_audio.empty()) {
        if (!WavWriter::Write(output_wav, all_audio, 24000)) {
            printf("ERROR: Failed to write WAV\n");
            return -1;
        }
        float total_audio_sec = all_audio.size() / 24000.0f;
        printf("\n========================================\n");
        printf("Long-text synthesis complete\n");
        printf("  Segments:        %zu\n", sentences.size());
        printf("  Audio duration:  %.2f s\n", total_audio_sec);
        printf("  NPU total:       %.3f s\n", total_npu_ms / 1000.0);
        printf("  Vocoder total:   %.3f s\n", total_vocoder_ms / 1000.0);
        printf("  RTF (NPU only):  %.4f\n", total_npu_ms / 1000.0 / total_audio_sec);
        printf("  RTF (end-to-end):%.4f\n", (total_npu_ms + total_vocoder_ms) / 1000.0 / total_audio_sec);
        printf("  Saved:           %s\n", output_wav.c_str());
        printf("========================================\n");
    }

    printf("\nDone!\n");
    return 0;
}
