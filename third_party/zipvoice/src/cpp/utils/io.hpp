/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Utility: AX engine I/O helpers
 *
 * Adapted from melotts.axera-main
 **************************************************************************************************/

#pragma once

#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#include "utils/checker.h"
#include "ax_sys_api.h"
#include "ax_engine_type.h"

#define IO_CMM_ALIGN_SIZE 128

namespace utils {

typedef enum {
    IO_BUFFER_STRATEGY_DEFAULT,
    IO_BUFFER_STRATEGY_CACHED
} IO_BUFFER_STRATEGY_T;

static inline void brief_io_info(const std::string& strModel, const AX_ENGINE_IO_INFO_T* io_info) {
    auto describe_shape_type = [](AX_ENGINE_TENSOR_LAYOUT_T type) -> const char* {
        switch (type) {
            case AX_ENGINE_TENSOR_LAYOUT_NHWC: return "NHWC";
            case AX_ENGINE_TENSOR_LAYOUT_NCHW: return "NCHW";
            default: return "unknown";
        }
    };
    auto describe_data_type = [](AX_ENGINE_DATA_TYPE_T type) -> const char* {
        switch (type) {
            case AX_ENGINE_DT_UINT8:  return "uint8";
            case AX_ENGINE_DT_UINT16: return "uint16";
            case AX_ENGINE_DT_FLOAT32: return "float32";
            case AX_ENGINE_DT_SINT16: return "sint16";
            case AX_ENGINE_DT_SINT8:  return "sint8";
            case AX_ENGINE_DT_SINT32: return "sint32";
            case AX_ENGINE_DT_UINT32: return "uint32";
            case AX_ENGINE_DT_FLOAT64: return "float64";
            default: return "unknown";
        }
    };

    printf("Model: %s\n", strModel.c_str());
    for (uint32_t i = 0; i < io_info->nInputSize; ++i) {
        auto& input = io_info->pInputs[i];
        printf("  Input[%d]: %s size=%u\n", i, input.pName, input.nSize);
    }
    for (uint32_t i = 0; i < io_info->nOutputSize; ++i) {
        auto& output = io_info->pOutputs[i];
        printf("  Output[%d]: %s size=%u\n", i, output.pName, output.nSize);
    }
}

static inline AX_S32 alloc_engine_buffer(const std::string& token, const std::string& appendix,
                                          size_t index, const AX_ENGINE_IOMETA_T* pMeta,
                                          AX_ENGINE_IO_BUFFER_T* pBuf,
                                          IO_BUFFER_STRATEGY_T eStrategy = IO_BUFFER_STRATEGY_DEFAULT) {
    memset(pBuf, 0, sizeof(AX_ENGINE_IO_BUFFER_T));
    pBuf->nSize = pMeta->nSize;

    const std::string token_name = "zipvoice_" + token + appendix + std::to_string(index);

    AX_S32 ret;
    if (eStrategy == IO_BUFFER_STRATEGY_CACHED) {
        ret = AX_SYS_MemAllocCached((AX_U64*)&pBuf->phyAddr, &pBuf->pVirAddr,
                                     pBuf->nSize, IO_CMM_ALIGN_SIZE,
                                     (const AX_S8*)token_name.c_str());
    } else {
        ret = AX_SYS_MemAlloc((AX_U64*)&pBuf->phyAddr, &pBuf->pVirAddr,
                               pBuf->nSize, IO_CMM_ALIGN_SIZE,
                               (const AX_S8*)token_name.c_str());
    }
    return ret;
}

static inline AX_S32 free_engine_buffer(AX_ENGINE_IO_BUFFER_T* pBuf) {
    if (pBuf->phyAddr == 0) {
        delete[] reinterpret_cast<uint8_t*>(pBuf->pVirAddr);
    } else {
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    pBuf->phyAddr = 0;
    pBuf->pVirAddr = nullptr;
    return 0;
}

static inline void free_io_index(AX_ENGINE_IO_BUFFER_T* io_buf, size_t index) {
    free_engine_buffer(io_buf + index);
}

static inline void free_io(AX_ENGINE_IO_T &io) {
    for (size_t j = 0; j < io.nInputSize; ++j) {
        free_io_index(io.pInputs, j);
    }
    for (size_t j = 0; j < io.nOutputSize; ++j) {
        free_io_index(io.pOutputs, j);
    }
    delete[] io.pInputs;
    delete[] io.pOutputs;
    io.pInputs = nullptr;
    io.pOutputs = nullptr;
}

static inline int prepare_io(const std::string& token, const AX_ENGINE_IO_INFO_T* info,
                              AX_ENGINE_IO_T &io, IO_BUFFER_STRATEGY_T strategy) {
    memset(&io, 0, sizeof(io));

    io.pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    if (!io.pInputs) return -1;
    memset(io.pInputs, 0x00, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);
    io.nInputSize = info->nInputSize;

    for (AX_U32 i = 0; i < info->nInputSize; ++i) {
        auto meta = info->pInputs[i];
        auto buffer = &io.pInputs[i];
        int ret = alloc_engine_buffer(token, "_input_", i, &meta, buffer, strategy);
        if (ret != 0) {
            for (AX_U32 j = 0; j < i; ++j) free_io_index(io.pInputs, j);
            delete[] io.pInputs;
            io.pInputs = nullptr;
            return ret;
        }
    }

    io.pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    if (!io.pOutputs) {
        delete[] io.pInputs;
        io.pInputs = nullptr;
        return -1;
    }
    memset(io.pOutputs, 0x00, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
    io.nOutputSize = info->nOutputSize;

    for (size_t i = 0; i < info->nOutputSize; ++i) {
        auto meta = info->pOutputs[i];
        auto buffer = &io.pOutputs[i];
        int ret = alloc_engine_buffer(token, "_output_", i, &meta, buffer, strategy);
        if (ret != 0) {
            for (AX_U32 j = 0; j < info->nInputSize; ++j) free_io_index(io.pInputs, j);
            delete[] io.pInputs;
            io.pInputs = nullptr;
            for (size_t k = 0; k < i; ++k) free_io_index(io.pOutputs, k);
            delete[] io.pOutputs;
            io.pOutputs = nullptr;
            return ret;
        }
    }
    return 0;
}

static inline AX_S32 push_io_input(void* input, int index, AX_ENGINE_IO_T& io) {
    AX_ENGINE_IO_BUFFER_T* pBuf = &io.pInputs[index];
    memcpy(pBuf->pVirAddr, input, pBuf->nSize);
    return 0;
}

static inline AX_S32 push_io_output(void* output, int index, AX_ENGINE_IO_T& io) {
    AX_ENGINE_IO_BUFFER_T* pBuf = &io.pOutputs[index];
    memcpy(output, pBuf->pVirAddr, pBuf->nSize);
    return 0;
}

static inline bool read_file(const char* path, std::vector<char>& data) {
    std::fstream fs(path, std::ios::in | std::ios::binary);
    if (!fs.is_open()) return false;

    fs.seekg(std::ios::end);
    auto fs_end = fs.tellg();
    fs.seekg(std::ios::beg);
    auto fs_beg = fs.tellg();

    auto file_size = static_cast<size_t>(fs_end - fs_beg);
    auto vector_size = data.size();
    data.reserve(vector_size + file_size);
    data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
    fs.close();
    return true;
}

static inline bool read_file(const char* path, AX_VOID **pModelBufferVirAddr,
                              AX_U64 &u64ModelBufferPhyAddr, AX_U32 &nModelBufferSize) {
    std::fstream fs(path, std::ios::in | std::ios::binary);
    if (!fs.is_open()) return false;

    fs.seekg(0, std::ios::end);
    int file_size = fs.tellg();
    fs.seekg(0, std::ios::beg);

    nModelBufferSize = (AX_U32)file_size;
    AX_SYS_MemAlloc(&u64ModelBufferPhyAddr, pModelBufferVirAddr, nModelBufferSize,
                     0x100, (AX_S8 *)"ZIPVOICE-MODEL");

    if (!pModelBufferVirAddr || (u64ModelBufferPhyAddr == 0)) return false;

    fs.read((AX_CHAR *)*pModelBufferVirAddr, nModelBufferSize);
    fs.close();
    return true;
}

} // namespace utils
