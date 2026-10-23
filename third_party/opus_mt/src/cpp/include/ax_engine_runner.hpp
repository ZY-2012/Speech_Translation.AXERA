#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Named multi-IO wrapper over AX Engine.
//
// OPUS-MT uses two graphs (encoder + decoder). The decoder has five inputs
// (three int32, two float32), so the fixed single-in/single-out helper does
// not apply. Buffers are allocated once and reused on every step.
class AxEngineRunner {
public:
    explicit AxEngineRunner(const std::string& model_path);
    ~AxEngineRunner();

    AxEngineRunner(const AxEngineRunner&) = delete;
    AxEngineRunner& operator=(const AxEngineRunner&) = delete;

    // Raw byte view of a named input/output. The caller knows the dtype and
    // shape from model_meta.json and memcpy's the matching array.
    void* input_data(const std::string& name);
    const void* output_data(const std::string& name) const;
    size_t input_bytes(const std::string& name) const;
    size_t output_bytes(const std::string& name) const;

    const std::vector<std::string>& input_names() const;
    const std::vector<std::string>& output_names() const;

    void run();

private:
    struct Impl;
    Impl* impl_;
};
