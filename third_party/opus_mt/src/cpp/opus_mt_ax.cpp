#include "ax_engine_runner.hpp"
#include "marian_tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* kPrefix = ">>cmn_Hans<< ";

struct Stats {
    double enc_ms = 0.0;
    double dec_ms = 0.0;
    double step_ms = 0.0;
    int src_tokens = 0;
    int out_tokens = 0;
    bool truncated = false;
};

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

class Translator {
public:
    Translator(const std::string& model_root, const std::string& direction, int max_new)
        : direction_(direction), max_new_(max_new) {
        const std::string dir = model_root + "/" + direction;
        tok_.reset(new MarianTokenizer(dir));
        enc_.reset(new AxEngineRunner(dir + "/encoder.axmodel"));
        dec_.reset(new AxEngineRunner(dir + "/decoder.axmodel"));
        if (max_new_ <= 0 || max_new_ > tok_->meta().tgt_len) max_new_ = tok_->meta().tgt_len;
        warmup();
    }

    std::string Translate(const std::string& input, Stats* stats) {
        std::string text = input;
        if (direction_ == "en-zh" && text.rfind(">>", 0) != 0) text = std::string(kPrefix) + text;

        bool truncated = false;
        std::vector<int32_t> ids = tok_->Encode(text, &truncated);
        const auto& meta = tok_->meta();

        std::vector<int32_t> enc_ids(static_cast<size_t>(meta.src_len), meta.pad);
        std::vector<float> enc_mask(static_cast<size_t>(meta.src_len), 0.0f);
        for (size_t i = 0; i < ids.size(); ++i) {
            enc_ids[i] = ids[i];
            enc_mask[i] = 1.0f;
        }

        std::memcpy(enc_->input_data("input_ids"), enc_ids.data(),
                    enc_ids.size() * sizeof(int32_t));
        std::memcpy(enc_->input_data("attention_mask"), enc_mask.data(),
                    enc_mask.size() * sizeof(float));

        const double t_enc0 = now_ms();
        enc_->run();
        const double t_enc1 = now_ms();

        const size_t hidden_bytes = static_cast<size_t>(meta.src_len) * 512 * sizeof(float);
        std::memcpy(dec_->input_data("encoder_hidden_states"),
                    enc_->output_data("encoder_hidden_states"), hidden_bytes);
        std::memcpy(dec_->input_data("encoder_attention_mask"), enc_mask.data(),
                    enc_mask.size() * sizeof(float));

        std::vector<int32_t> out;
        out.reserve(static_cast<size_t>(max_new_) + 1);
        out.push_back(meta.start);
        std::vector<int32_t> dec_ids(static_cast<size_t>(meta.tgt_len), meta.pad);
        std::vector<int32_t> dec_mask(static_cast<size_t>(meta.tgt_len), 0);
        std::vector<int32_t> pos_ids(static_cast<size_t>(meta.tgt_len), 0);

        double dec_total = 0.0;
        for (int step = 0; step < max_new_; ++step) {
            const int t = static_cast<int>(out.size());
            std::fill(dec_ids.begin(), dec_ids.end(), meta.pad);
            std::fill(dec_mask.begin(), dec_mask.end(), 0);
            std::fill(pos_ids.begin(), pos_ids.end(), 0);
            const int offset = meta.tgt_len - t;
            for (int i = 0; i < t; ++i) {
                dec_ids[offset + i] = out[i];
                dec_mask[offset + i] = 1;
                pos_ids[offset + i] = i;
            }
            std::memcpy(dec_->input_data("decoder_input_ids"), dec_ids.data(),
                        dec_ids.size() * sizeof(int32_t));
            std::memcpy(dec_->input_data("decoder_attention_mask"), dec_mask.data(),
                        dec_mask.size() * sizeof(int32_t));
            std::memcpy(dec_->input_data("position_ids"), pos_ids.data(),
                        pos_ids.size() * sizeof(int32_t));

            const double s0 = now_ms();
            dec_->run();
            const double s1 = now_ms();
            dec_total += (s1 - s0);

            const float* logits = static_cast<const float*>(dec_->output_data("logits"));
            int best = 0;
            float best_val = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < meta.vocab_size; ++i) {
                if (i == meta.pad) continue;
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best = i;
                }
            }
            out.push_back(best);
            if (best == meta.eos) break;
        }

        std::vector<int32_t> body(out.begin() + 1, out.end());
        if (stats) {
            stats->enc_ms = t_enc1 - t_enc0;
            stats->dec_ms = dec_total;
            stats->step_ms = dec_total / std::max(1, static_cast<int>(body.size()));
            stats->src_tokens = static_cast<int>(enc_mask_sum(enc_mask));
            stats->out_tokens = static_cast<int>(body.size());
            stats->truncated = truncated;
        }
        return tok_->Decode(body);
    }

private:
    static int enc_mask_sum(const std::vector<float>& mask) {
        int n = 0;
        for (float v : mask) n += (v > 0.5f) ? 1 : 0;
        return n;
    }

    void warmup() {
        for (int i = 0; i < 2; ++i) {
            try {
                Translate(direction_ == "zh-en" ? "你好" : "Hello", nullptr);
            } catch (...) {
            }
        }
    }

    std::string direction_;
    int max_new_ = 64;
    std::unique_ptr<MarianTokenizer> tok_;
    std::unique_ptr<AxEngineRunner> enc_;
    std::unique_ptr<AxEngineRunner> dec_;
};

void print_stats(const Stats& s) {
    std::cout << "enc " << s.enc_ms << " ms | dec " << s.dec_ms << " ms | step "
              << s.step_ms << " ms | src_tokens " << s.src_tokens << " | out_tokens "
              << s.out_tokens << " | truncated " << (s.truncated ? "yes" : "no") << "\n";
}
}  // namespace

int main(int argc, char** argv) {
    std::string direction, model_root = "models", text, input, output;
    int max_new = 64;
    bool stats = false;
    bool has_text = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--direction") direction = need("--direction");
        else if (a == "--model-root") model_root = need("--model-root");
        else if (a == "--text") { text = need("--text"); has_text = true; }
        else if (a == "--input") input = need("--input");
        else if (a == "--output") output = need("--output");
        else if (a == "--max-new-tokens") max_new = std::stoi(need("--max-new-tokens"));
        else if (a == "--stats") stats = true;
        else if (a == "-h" || a == "--help") {
            std::cout << "usage: " << argv[0]
                      << " --direction zh-en|en-zh [--model-root models] "
                         "[--text TEXT | --input FILE [--output FILE]] "
                         "[--max-new-tokens 64] [--stats]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }
    if (direction != "zh-en" && direction != "en-zh") {
        std::cerr << "--direction must be zh-en or en-zh\n";
        return 2;
    }
    try {
        Translator tr(model_root, direction, max_new);
        if (has_text) {
            Stats s;
            std::cout << tr.Translate(text, &s) << "\n";
            if (stats) print_stats(s);
            return 0;
        }
        if (input.empty()) {
            std::cerr << "either --text or --input is required\n";
            return 2;
        }
        std::ifstream fin(input);
        if (!fin) throw std::runtime_error("cannot open input: " + input);
        std::vector<std::string> lines, outs;
        std::string line;
        while (std::getline(fin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.find_first_not_of(" \t") == std::string::npos) continue;
            lines.push_back(line);
        }
        Stats total;
        const double t0 = now_ms();
        for (const auto& l : lines) {
            Stats s;
            outs.push_back(tr.Translate(l, &s));
            total.enc_ms += s.enc_ms;
            total.dec_ms += s.dec_ms;
            total.out_tokens += s.out_tokens;
            total.src_tokens += s.src_tokens;
            total.truncated = total.truncated || s.truncated;
        }
        const double dt = now_ms() - t0;
        if (!output.empty()) {
            std::ofstream fout(output);
            if (!fout) throw std::runtime_error("cannot open output: " + output);
            for (const auto& o : outs) fout << o << "\n";
        } else {
            for (const auto& o : outs) std::cout << o << "\n";
        }
        const int n = std::max(1, static_cast<int>(outs.size()));
        std::cout << "translated " << outs.size() << " lines in " << dt / 1000.0
                  << "s (" << dt / n << " ms/line)\n";
        if (stats) {
            std::cout << "avg encoder " << total.enc_ms / n << " ms | decoder "
                      << total.dec_ms / n << " ms | step " << total.dec_ms / std::max(1, total.out_tokens)
                      << " ms | tokens/line " << static_cast<double>(total.out_tokens) / n << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
