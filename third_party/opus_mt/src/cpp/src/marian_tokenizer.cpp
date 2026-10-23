#include "marian_tokenizer.hpp"

#include "sentencepiece_processor.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace {
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string trim(std::string s) {
    const char* ws = " \t\r\n";
    const auto begin = s.find_first_not_of(ws);
    if (begin == std::string::npos) return "";
    const auto end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1);
}
}  // namespace

MarianTokenizer::MarianTokenizer(const std::string& model_dir) {
    const std::string dir = model_dir;
    auto vocab = nlohmann::json::parse(read_file(dir + "/vocab.json"), nullptr, false);
    if (vocab.is_discarded()) throw std::runtime_error("invalid vocab.json");

    int max_id = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const int id = it.value().get<int>();
        piece_to_id_[it.key()] = id;
        max_id = std::max(max_id, id);
    }
    id_to_piece_.assign(static_cast<size_t>(max_id) + 1, std::string());
    for (const auto& kv : piece_to_id_) id_to_piece_[kv.second] = kv.first;

    auto meta = nlohmann::json::parse(read_file(dir + "/model_meta.json"), nullptr, false);
    if (meta.is_discarded()) throw std::runtime_error("invalid model_meta.json");
    meta_.src_len = meta.value("src_len", 128);
    meta_.tgt_len = meta.value("tgt_len", 64);
    meta_.pad = meta.value("pad_token_id", 65000);
    meta_.eos = meta.value("eos_token_id", 0);
    meta_.start = meta.value("decoder_start_token_id", 65000);
    meta_.vocab_size = meta.value("vocab_size", static_cast<int>(id_to_piece_.size()));

    auto src = piece_to_id_.find("<unk>");
    meta_.unk = (src == piece_to_id_.end()) ? 1 : src->second;

    spm_src_.reset(new sentencepiece::SentencePieceProcessor());
    spm_tgt_.reset(new sentencepiece::SentencePieceProcessor());
    if (!spm_src_->Load(dir + "/source.spm").ok())
        throw std::runtime_error("cannot load source.spm");
    if (!spm_tgt_->Load(dir + "/target.spm").ok())
        throw std::runtime_error("cannot load target.spm");
}

MarianTokenizer::~MarianTokenizer() = default;

int32_t MarianTokenizer::lookup(const std::string& piece) const {
    auto it = piece_to_id_.find(piece);
    return it == piece_to_id_.end() ? static_cast<int32_t>(meta_.unk) : it->second;
}

std::vector<int32_t> MarianTokenizer::Encode(const std::string& text, bool* truncated) const {
    std::string code;
    std::string rest = text;
    if (text.rfind(">>", 0) == 0) {
        const auto end = text.find("<<");
        if (end != std::string::npos) {
            code = text.substr(0, end + 2);
            rest = text.substr(end + 2);
        }
    }

    std::vector<std::string> pieces;
    if (!rest.empty()) {
        const auto status = spm_src_->Encode(rest, &pieces);
        if (!status.ok()) throw std::runtime_error("SentencePiece encode failed");
    }

    std::vector<int32_t> ids;
    ids.reserve(pieces.size() + 2);
    if (!code.empty()) ids.push_back(lookup(code));
    for (const auto& p : pieces) ids.push_back(lookup(p));

    const int limit = meta_.src_len - 1;
    if (truncated) *truncated = static_cast<int>(ids.size()) > limit;
    if (static_cast<int>(ids.size()) > limit) ids.resize(limit);
    ids.push_back(static_cast<int32_t>(meta_.eos));
    return ids;
}

std::string MarianTokenizer::Decode(const std::vector<int32_t>& ids) const {
    static const std::string kUnderline = "\xE2\x96\x81";  // U+2581
    std::vector<std::string> pieces;
    pieces.reserve(ids.size());
    for (int32_t id : ids) {
        if (id == meta_.eos || id == meta_.pad || id == meta_.unk) continue;
        if (id < 0 || id >= static_cast<int32_t>(id_to_piece_.size())) continue;
        const std::string& p = id_to_piece_[id];
        if (p.empty()) continue;
        if (p.size() >= 4 && p.compare(0, 2, ">>") == 0 && p.compare(p.size() - 2, 2, "<<") == 0)
            continue;
        pieces.push_back(p);
    }
    std::string text = spm_src_->DecodePieces(pieces);
    size_t pos = 0;
    while ((pos = text.find(kUnderline, pos)) != std::string::npos) {
        text.replace(pos, kUnderline.size(), " ");
        pos += 1;
    }
    return trim(text);
}
