/**************************************************************************************************
 * ZipVoice AXCL C++ Port
 *
 * EngineWrapper implementation using middleware::runner (AXCL Runtime API).
 *
 * Key differences from AXERA:
 *   - Uses axclrtMemcpy for host↔device data transfer
 *   - No CMM (contiguous memory) allocation
 *   - Runner manages device-side memory and NPU execution
 **************************************************************************************************/

#include "EngineWrapper.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <axcl.h>

EngineWrapper::EngineWrapper()
    : m_hasInit(false), m_input_num(0), m_output_num(0) {}

EngineWrapper::~EngineWrapper() {
    Release();
}

int EngineWrapper::Init(const char* strModelPath,
                         uint32_t nNpuType,
                         const char* axclConfig) {
    const char* config = axclConfig ? axclConfig : "/usr/local/axcl/axcl.json";
    uint32_t device_index = nNpuType;      // nNpuType = device_index on AXCL
    uint32_t npu_kind = 0;                  // AXCL_VNPU_DISABLE (default)

    // 1. Create runtime runner
    m_runner = std::make_unique<middleware::runtime_runner>();

    // 2. Initialize runner (handles axclrtEngineInit internally)
    if (!m_runner->init(config, device_index, npu_kind)) {
        fprintf(stderr, "[ERROR] AXCL runner init failed for model: %s\n", strModelPath);
        return -1;
    }

    // 3. Load model from file
    if (!m_runner->load(strModelPath)) {
        fprintf(stderr, "[ERROR] AXCL runner load failed for model: %s\n", strModelPath);
        return -1;
    }

    // 4. Prepare IO buffers (allocates device memory, sets up IO)
    if (!m_runner->prepare(true, true, 0, 0)) {
        fprintf(stderr, "[ERROR] AXCL runner prepare failed for model: %s\n", strModelPath);
        return -1;
    }

    // 5. Build name-to-index maps
    m_input_num = static_cast<int>(m_runner->get_input_count());
    m_output_num = static_cast<int>(m_runner->get_output_count());

    m_input_name_to_idx.clear();
    for (int i = 0; i < m_input_num; ++i) {
        std::string name = m_runner->get_input_name(i);
        m_input_name_to_idx[name] = i;
    }

    m_output_name_to_idx.clear();
    for (int i = 0; i < m_output_num; ++i) {
        std::string name = m_runner->get_output_name(i);
        m_output_name_to_idx[name] = i;
    }

    m_hasInit = true;

    printf("AXCL EngineWrapper loaded: %s (inputs=%d, outputs=%d)\n",
           strModelPath, m_input_num, m_output_num);
    for (int i = 0; i < m_input_num; ++i) {
        printf("  input[%d]: %s (%zu bytes)\n", i,
               m_runner->get_input_name(i).c_str(),
               (size_t)m_runner->get_input_size(i));
    }
    for (int i = 0; i < m_output_num; ++i) {
        printf("  output[%d]: %s (%zu bytes)\n", i,
               m_runner->get_output_name(i).c_str(),
               (size_t)m_runner->get_output_size(i));
    }

    return 0;
}

int EngineWrapper::SetInput(void* pInput, int index) {
    if (!m_hasInit || index < 0 || index >= m_input_num) return -1;

    uintmax_t size = m_runner->get_input_size(index);
    void* dev_ptr = m_runner->get_input_pointer(index);

    if (!dev_ptr || size == 0) {
        fprintf(stderr, "[ERROR] Invalid input pointer/size for index %d\n", index);
        return -1;
    }

    // Copy host data to device memory
    axclError ret = axclrtMemcpy(dev_ptr, pInput, size, AXCL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        fprintf(stderr, "[ERROR] axclrtMemcpy H2D failed for input %d: 0x%x\n", index, ret);
        return -1;
    }

    return 0;
}

int EngineWrapper::RunSync() {
    if (!m_hasInit) return -1;

    if (!m_runner->run(false)) {
        fprintf(stderr, "[ERROR] AXCL runner run failed\n");
        return -1;
    }

    return 0;
}

int EngineWrapper::GetOutput(void* pOutput, int index) {
    if (!m_hasInit || index < 0 || index >= m_output_num) return -1;

    uintmax_t size = m_runner->get_output_size(index);
    void* dev_ptr = m_runner->get_output_pointer(index);

    if (!dev_ptr || size == 0) {
        fprintf(stderr, "[ERROR] Invalid output pointer/size for index %d\n", index);
        return -1;
    }

    // Copy device data to host memory
    axclError ret = axclrtMemcpy(pOutput, dev_ptr, size, AXCL_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        fprintf(stderr, "[ERROR] axclrtMemcpy D2H failed for output %d: 0x%x\n", index, ret);
        return -1;
    }

    return 0;
}

int EngineWrapper::SetInputByName(const char* name, void* pInput) {
    auto it = m_input_name_to_idx.find(std::string(name));
    if (it == m_input_name_to_idx.end()) {
        fprintf(stderr, "[ERROR] Input '%s' not found in model\n", name);
        return -1;
    }
    return SetInput(pInput, it->second);
}

int EngineWrapper::GetOutputByName(const char* name, void* pOutput) {
    auto it = m_output_name_to_idx.find(std::string(name));
    if (it == m_output_name_to_idx.end()) {
        fprintf(stderr, "[ERROR] Output '%s' not found in model\n", name);
        return -1;
    }
    return GetOutput(pOutput, it->second);
}

int EngineWrapper::GetInputSize(int index) {
    if (!m_hasInit || index < 0 || index >= m_input_num) return -1;
    return static_cast<int>(m_runner->get_input_size(index));
}

int EngineWrapper::GetOutputSize(int index) {
    if (!m_hasInit || index < 0 || index >= m_output_num) return -1;
    return static_cast<int>(m_runner->get_output_size(index));
}

int EngineWrapper::GetInputSizeByName(const char* name) {
    int idx = GetInputIndex(name);
    if (idx < 0) return -1;
    return GetInputSize(idx);
}

int EngineWrapper::GetOutputSizeByName(const char* name) {
    int idx = GetOutputIndex(name);
    if (idx < 0) return -1;
    return GetOutputSize(idx);
}

int EngineWrapper::GetInputIndex(const char* name) {
    auto it = m_input_name_to_idx.find(std::string(name));
    return (it != m_input_name_to_idx.end()) ? it->second : -1;
}

int EngineWrapper::GetOutputIndex(const char* name) {
    auto it = m_output_name_to_idx.find(std::string(name));
    return (it != m_output_name_to_idx.end()) ? it->second : -1;
}

int EngineWrapper::Release() {
    if (m_runner) {
        m_runner->final();
        m_runner.reset();
    }
    m_hasInit = false;
    m_input_name_to_idx.clear();
    m_output_name_to_idx.clear();
    m_input_num = 0;
    m_output_num = 0;
    return 0;
}

const char* EngineWrapper::GetInputName(int index) const {
    static thread_local std::string cached_name;
    if (!m_hasInit || index < 0 || index >= m_input_num) return nullptr;
    cached_name = m_runner->get_input_name(index);
    return cached_name.c_str();
}

const char* EngineWrapper::GetOutputName(int index) const {
    static thread_local std::string cached_name;
    if (!m_hasInit || index < 0 || index >= m_output_num) return nullptr;
    cached_name = m_runner->get_output_name(index);
    return cached_name.c_str();
}
