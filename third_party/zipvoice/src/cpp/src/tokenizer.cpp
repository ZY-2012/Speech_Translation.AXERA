/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Tokenizer implementation. Uses pinyin_table.hpp (auto-generated from pypinyin.pinyin_dict)
 * for Chinese character → pinyin token mapping, matching Python LocalEmiliaTokenizer output.
 **************************************************************************************************/

#include "tokenizer.hpp"
#include "src/pinyin_table.hpp"

#include <fstream>
#include <sstream>
#include <regex>
#include <cctype>
#include <algorithm>

Tokenizer::Tokenizer() : m_pad_id(0), m_loaded(false) {}

int Tokenizer::Load(const std::string& token_file) {
    std::ifstream file(token_file);
    if (!file.is_open()) {
        printf("Failed to open token file: %s\n", token_file.c_str());
        return -1;
    }

    m_token2id.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // Format: token\tid
        size_t tab_pos = line.rfind('\t');
        if (tab_pos == std::string::npos) continue;

        std::string token = line.substr(0, tab_pos);
        int id = std::stoi(line.substr(tab_pos + 1));
        m_token2id[token] = id;
    }
    file.close();

    // Build reverse mapping
    int max_id = 0;
    for (auto& kv : m_token2id) {
        if (kv.second > max_id) max_id = kv.second;
    }
    m_id2token.resize(max_id + 1);
    for (auto& kv : m_token2id) {
        m_id2token[kv.second] = kv.first;
    }

    auto it = m_token2id.find("_");
    if (it != m_token2id.end()) {
        m_pad_id = it->second;
    } else {
        m_pad_id = 0;
        printf("WARNING: pad token '_' not found in token file, using 0 as pad_id\n");
    }

    m_loaded = true;
    printf("Loaded %zu tokens from %s, pad_id=%d\n", m_token2id.size(), token_file.c_str(), m_pad_id);
    return 0;
}

int Tokenizer::TokenToId(const std::string& token) const {
    auto it = m_token2id.find(token);
    return (it != m_token2id.end()) ? it->second : m_pad_id;
}

bool Tokenizer::IsChinese(char c) {
    return (c >= 0x4E00 && c <= 0x9FFF);
}

bool Tokenizer::IsAlphabet(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

std::string Tokenizer::MapPunctuation(const std::string& text) {
    // Map full-width Chinese punctuation to ASCII equivalents.
    // UTF-8 code points:
    //   \xEF\xBC\x8C = U+FF0C (，)  \xEF\xBC\x9A = U+FF1A (：)
    //   \xE3\x80\x82 = U+3002 (。)  \xE3\x80\x81 = U+3001 (、)
    //   \xEF\xBC\x81 = U+FF01 (！)  \xE2\x80\x9C = U+201C (")
    //   \xEF\xBC\x9F = U+FF1F (？)  \xE2\x80\x9D = U+201D (")
    //   \xEF\xBC\x9B = U+FF1B (；)  \xE2\x80\x98 = U+2018 (')
    //   \xE2\x80\xA6 = U+2026 (…)   \xE2\x80\x99 = U+2019 (')
    //   \xE2\x8B\xAF = U+22EF (⋯)
    static const std::pair<std::string, std::string> replacements[] = {
        {"\xEF\xBC\x8C", ","},   // ，
        {"\xE3\x80\x82", "."},   // 。
        {"\xEF\xBC\x81", "!"},   // ！
        {"\xEF\xBC\x9F", "?"},   // ？
        {"\xEF\xBC\x9B", ";"},   // ；
        {"\xEF\xBC\x9A", ":"},   // ：
        {"\xE3\x80\x81", ","},   // 、
        {"\xE2\x80\x9C", "\""},  // "
        {"\xE2\x80\x9D", "\""},  // "
        {"\xE2\x80\x98", "'"},   // '
        {"\xE2\x80\x99", "'"},   // '
        {"\xE2\x80\xA6", "..."}, // …
        {"\xE2\x8B\xAF", "..."}, // ⋯
    };
    std::string result = text;
    for (auto& rep : replacements) {
        size_t pos = 0;
        while ((pos = result.find(rep.first, pos)) != std::string::npos) {
            result.replace(pos, rep.first.length(), rep.second);
            pos += rep.second.length();
        }
    }
    return result;
}

std::vector<std::pair<std::string, std::string>>
Tokenizer::GetSegments(const std::string& text) const {
    std::vector<std::pair<std::string, std::string>> segments;
    if (text.empty()) return segments;

    std::string current;
    std::string current_lang;

    auto char_lang = [](char c) -> std::string {
        if (IsChinese(c)) return "zh";
        if (IsAlphabet(c)) return "en";
        return "other";
    };

    for (size_t i = 0; i < text.size(); ) {
        // Handle UTF-8 multi-byte characters
        size_t char_len = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        std::string ch = text.substr(i, char_len);
        std::string lang;

        if (char_len > 1) {
            lang = "zh"; // Assume multi-byte UTF-8 is Chinese
        } else if (IsAlphabet(ch[0])) {
            lang = "en";
        } else {
            lang = "other";
        }

        if (current.empty()) {
            current = ch;
            current_lang = lang;
        } else if (current_lang == "other" || lang == "other" || current_lang == lang) {
            current += ch;
            if (current_lang == "other" && lang != "other") current_lang = lang;
        } else {
            segments.push_back({current, current_lang});
            current = ch;
            current_lang = lang;
        }
        i += char_len;
    }
    if (!current.empty()) {
        segments.push_back({current, current_lang});
    }
    return segments;
}

std::vector<std::string> Tokenizer::TokenizeZh(const std::string& text) const {
    // Character-by-character pinyin tokenization using pinyin_table.hpp.
    // Matches Python LocalEmiliaTokenizer (without jieba word segmentation).
    std::vector<std::string> tokens;
    std::string normalized = MapPunctuation(text);

    for (size_t i = 0; i < normalized.size(); ) {
        unsigned char c = static_cast<unsigned char>(normalized[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        std::string ch = normalized.substr(i, char_len);
        i += char_len;

        // Punctuation or ASCII
        if (char_len == 1) {
            std::string mapped = MapPunctuation(ch);
            if (mapped != ch && m_token2id.find(mapped) != m_token2id.end()) {
                tokens.push_back(mapped);
                continue;
            }
            if (m_token2id.find(ch) != m_token2id.end()) {
                tokens.push_back(ch);
                continue;
            }
        }

        // Chinese character: lookup pinyin token IDs via generated table
        if (char_len == 3) {
            uint32_t cp = ((unsigned char)ch[0] & 0x0F) << 12
                        | ((unsigned char)ch[1] & 0x3F) << 6
                        | ((unsigned char)ch[2] & 0x3F);
            auto ids = pinyin_lookup(cp);
            for (int tid : ids) {
                if (tid >= 0 && tid < (int)m_id2token.size() && !m_id2token[tid].empty()) {
                    tokens.push_back(m_id2token[tid]);
                }
            }
        }
    }
    return tokens;
}

std::vector<std::string> Tokenizer::TokenizeEn(const std::string& text) const {
    // Simplified English tokenization: character-by-character phoneme mapping
    // In production, use the Python tokenizer with piper_phonemize
    std::vector<std::string> tokens;
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    lower = MapPunctuation(lower);

    for (size_t i = 0; i < lower.size(); ++i) {
        char c = lower[i];
        if (c == ' ') continue;
        if (ispunct(c)) {
            // Map common punctuation
            switch (c) {
                case ',': tokens.push_back(","); break;
                case '.': tokens.push_back("."); break;
                case '!': tokens.push_back("!"); break;
                case '?': tokens.push_back("?"); break;
                case ';': tokens.push_back(";"); break;
                case ':': tokens.push_back(":"); break;
                case '\'': tokens.push_back("'"); break;
                case '"': tokens.push_back("\""); break;
                default: break;
            }
            continue;
        }
        // Output character as-is (phoneme-based tokens)
        std::string token(1, c);
        if (m_token2id.find(token) != m_token2id.end()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<int> Tokenizer::TextToTokenIds(const std::string& text) const {
    if (!m_loaded) return {};

    std::string normalized = MapPunctuation(text);
    auto segments = GetSegments(normalized);

    std::vector<int> token_ids;
    for (auto& seg : segments) {
        std::vector<std::string> tokens;
        if (seg.second == "zh") {
            tokens = TokenizeZh(seg.first);
        } else if (seg.second == "en") {
            tokens = TokenizeEn(seg.first);
        }
        // "other" segments: try individual chars as tokens
        else {
            for (char c : seg.first) {
                std::string token(1, c);
                if (m_token2id.find(token) != m_token2id.end()) {
                    token_ids.push_back(m_token2id.at(token));
                }
            }
            continue;
        }

        for (auto& token : tokens) {
            auto it = m_token2id.find(token);
            if (it != m_token2id.end()) {
                token_ids.push_back(it->second);
            }
        }
    }
    return token_ids;
}

std::vector<std::vector<int>> Tokenizer::TextsToTokenIds(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> result;
    result.reserve(texts.size());
    for (auto& text : texts) {
        result.push_back(TextToTokenIds(text));
    }
    return result;
}

void Tokenizer::BuildCatTokens(const std::vector<int>& prompt_tokens,
                                const std::vector<int>& text_tokens,
                                int max_tokens,
                                std::vector<int32_t>& out_cat_tokens) const {
    out_cat_tokens.assign(max_tokens, static_cast<int32_t>(m_pad_id));

    size_t pos = 0;
    for (size_t i = 0; i < prompt_tokens.size() && pos < (size_t)max_tokens; ++i) {
        out_cat_tokens[pos++] = static_cast<int32_t>(prompt_tokens[i]);
    }
    for (size_t i = 0; i < text_tokens.size() && pos < (size_t)max_tokens; ++i) {
        out_cat_tokens[pos++] = static_cast<int32_t>(text_tokens[i]);
    }
    if (pos < (size_t)max_tokens) {
        out_cat_tokens[pos] = static_cast<int32_t>(m_pad_id);
    }
}
