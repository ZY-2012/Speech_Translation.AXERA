#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sentencepiece {
class SentencePieceProcessor;
}

// Marian tokenizer used by Helsinki-NLP/opus-mt-*:
//   source text -> source.spm pieces -> shared vocab.json ids -> + </s>
//   model ids  -> reversed vocab pieces -> source.spm DecodePieces
// The >>lang<< prefix is split before SentencePiece, matching MarianTokenizer.
class MarianTokenizer {
public:
    struct Meta {
        int src_len = 128;
        int tgt_len = 64;
        int pad = 65000;
        int eos = 0;
        int start = 65000;
        int unk = 1;
        int vocab_size = 65001;
    };

    explicit MarianTokenizer(const std::string& model_dir);
    ~MarianTokenizer();

    std::vector<int32_t> Encode(const std::string& text, bool* truncated = nullptr) const;
    std::string Decode(const std::vector<int32_t>& ids) const;
    const Meta& meta() const { return meta_; }

private:
    int32_t lookup(const std::string& piece) const;

    std::unique_ptr<sentencepiece::SentencePieceProcessor> spm_src_;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spm_tgt_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::vector<std::string> id_to_piece_;
    Meta meta_;
};
