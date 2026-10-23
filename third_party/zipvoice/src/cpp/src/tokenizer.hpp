/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Tokenizer: loads tokens.txt and maps token strings to IDs.
 *
 * For board deployment, tokenization should ideally be done on a host machine
 * and token IDs passed directly. This class provides basic tokenization
 * capability for convenience.
 **************************************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Tokenizer {
public:
    Tokenizer();

    /**
     * Load tokens.txt file. Format: token\tid per line.
     */
    int Load(const std::string& token_file);

    /**
     * Convert a single text to token IDs using simplified rules.
     * For production use, pre-compute token IDs with the Python tokenizer.
     */
    std::vector<int> TextToTokenIds(const std::string& text) const;

    /**
     * Convert multiple texts to token IDs.
     */
    std::vector<std::vector<int>> TextsToTokenIds(const std::vector<std::string>& texts) const;

    /**
     * Get the pad token ID ("_" token).
     */
    int GetPadId() const { return m_pad_id; }

    /**
     * Get vocabulary size.
     */
    int GetVocabSize() const { return static_cast<int>(m_token2id.size()); }

    /**
     * Lookup a single token. Returns pad_id if not found.
     */
    int TokenToId(const std::string& token) const;

    /**
     * Build concatenated token array [prompt_tokens + text_tokens + pad] padded to max_tokens.
     */
    void BuildCatTokens(const std::vector<int>& prompt_tokens,
                        const std::vector<int>& text_tokens,
                        int max_tokens,
                        std::vector<int32_t>& out_cat_tokens) const;

    bool IsLoaded() const { return m_loaded; }

private:
    std::unordered_map<std::string, int> m_token2id;
    std::vector<std::string> m_id2token;
    int m_pad_id;
    bool m_loaded;

    // Simplified text→tokens for Chinese (char-by-char pinyin-like tokens)
    std::vector<std::string> TokenizeZh(const std::string& text) const;
    // Simplified text→tokens for English (basic phoneme rules)
    std::vector<std::string> TokenizeEn(const std::string& text) const;
    // Split text into language segments
    std::vector<std::pair<std::string, std::string>> GetSegments(const std::string& text) const;

    static bool IsChinese(char c);
    static bool IsAlphabet(char c);
    static std::string MapPunctuation(const std::string& text);
};
