/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * EngineWrapper implementation
 **************************************************************************************************/

#include "EngineWrapper.hpp"
#include "utils/io.hpp"

#include <cstdlib>

#if !defined(AX630C)
static const char *strAlgoModelType[AX_ENGINE_MODEL_TYPE_BUTT] = {"3.6T", "7.2T", "10.8T"};
#endif

// NPU type enum
typedef enum axNPU_TYPE_E {
    AX_NPU_DEFAULT = 0,
    AX_STD_VNPU_1 = (1 << 0),
    AX_STD_VNPU_2 = (1 << 1),
    AX_STD_VNPU_3 = (1 << 2),
    AX_BL_VNPU_1  = (1 << 3),
    AX_BL_VNPU_2  = (1 << 4)
} AX_NPU_TYPE_E;

#if !defined(AX630C)
static AX_S32 CheckModelVNpu(const std::string &strModel,
                              const AX_ENGINE_MODEL_TYPE_T &eModelType,
                              const AX_S32 &nNpuType, AX_U32 &nNpuSet) {
    AX_ENGINE_NPU_ATTR_T stNpuAttr;
    memset(&stNpuAttr, 0x00, sizeof(stNpuAttr));

    auto ret = AX_ENGINE_GetVNPUAttr(&stNpuAttr);
    if (ret == 0) {
        if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_DISABLE) {
            nNpuSet = 0x01;
        } else if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_STD) {
            if (eModelType == AX_ENGINE_MODEL_TYPE1 || eModelType == AX_ENGINE_MODEL_TYPE2)
                return -1;
            if (nNpuType == 0) nNpuSet = 0x02;
            else {
                if (nNpuType & AX_STD_VNPU_1) nNpuSet |= 0x01;
                if (nNpuType & AX_STD_VNPU_2) nNpuSet |= 0x02;
                if (nNpuType & AX_STD_VNPU_3) nNpuSet |= 0x04;
            }
        } else if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE) {
            if (eModelType == AX_ENGINE_MODEL_TYPE2) return -1;
            if (nNpuType == 0) {
                nNpuSet = (eModelType == AX_ENGINE_MODEL_TYPE1) ? 0x01 : 0x02;
            } else {
                if (eModelType == AX_ENGINE_MODEL_TYPE1) {
                    if (nNpuType & AX_BL_VNPU_2) return -1;
                    if (nNpuType & AX_BL_VNPU_1) nNpuSet |= 0x01;
                } else {
                    if (nNpuType & AX_BL_VNPU_1) nNpuSet |= 0x01;
                    if (nNpuType & AX_BL_VNPU_2) nNpuSet |= 0x02;
                }
            }
        }
    }
    return ret;
}
#endif

EngineWrapper::EngineWrapper()
    : m_hasInit(false), m_handle(nullptr), m_io_info(nullptr),
      m_input_num(0), m_output_num(0) {
    memset(&m_io, 0, sizeof(m_io));
}

EngineWrapper::~EngineWrapper() {
    Release();
}

int EngineWrapper::Init(const char* strModelPath, uint32_t nNpuType, const char* axclConfig) {
    (void)axclConfig;  // unused on AXERA
    // 1. Load model
    AX_BOOL bLoadModelUseCmm = AX_TRUE;
    AX_CHAR *pModelBufferVirAddr = nullptr;
    AX_U64 u64ModelBufferPhyAddr = 0;
    AX_U32 nModelBufferSize = 0;

    if (bLoadModelUseCmm) {
        if (!utils::read_file(strModelPath, (AX_VOID **)&pModelBufferVirAddr,
                               u64ModelBufferPhyAddr, nModelBufferSize)) {
            printf("Read model(%s) fail\n", strModelPath);
            return -1;
        }
    } else {
        std::vector<char> model_buffer;
        if (!utils::read_file(strModelPath, model_buffer)) {
            printf("Read model(%s) fail\n", strModelPath);
            return -1;
        }
        pModelBufferVirAddr = model_buffer.data();
        nModelBufferSize = model_buffer.size();
    }

    auto freeModelBuffer = [&]() {
        if (bLoadModelUseCmm) {
            if (u64ModelBufferPhyAddr != 0)
                AX_SYS_MemFree(u64ModelBufferPhyAddr, &pModelBufferVirAddr);
        }
    };

    // 1.1 Get Model Type & Check VNPU (AX650 only)
#if !defined(AX630C)
    AX_ENGINE_MODEL_TYPE_T eModelType = AX_ENGINE_MODEL_TYPE0;
    AX_S32 ret = AX_ENGINE_GetModelType(pModelBufferVirAddr, nModelBufferSize, &eModelType);
    if (0 != ret || eModelType >= AX_ENGINE_MODEL_TYPE_BUTT) {
        printf("%s AX_ENGINE_GetModelType fail ret=%x\n", strModelPath, ret);
        freeModelBuffer();
        return -1;
    }

    AX_U32 nNpuSet = 0;
    ret = CheckModelVNpu(strModelPath, eModelType, nNpuType, nNpuSet);
    if (0 != ret) {
        printf("CheckModelVNpu fail\n");
        freeModelBuffer();
        return -1;
    }
#endif
#if defined(AX630C)
    AX_S32 ret = 0;
#endif

    // 2. Create handle
    AX_ENGINE_HANDLE handle = nullptr;
    ret = AX_ENGINE_CreateHandle(&handle, pModelBufferVirAddr, nModelBufferSize);
    freeModelBuffer();

    auto deinit_handle = [&handle]() {
        if (handle) { AX_ENGINE_DestroyHandle(handle); }
        return -1;
    };

    if (0 != ret || !handle) {
        printf("Create model(%s) handle fail\n", strModelPath);
        return deinit_handle();
    }

    // 3. Create context
    ret = AX_ENGINE_CreateContext(handle);
    if (0 != ret) return deinit_handle();

    // 4. Get IO info
    m_io_info = nullptr;
    ret = AX_ENGINE_GetIOInfo(handle, &m_io_info);
    if (0 != ret) return deinit_handle();

    m_input_num = m_io_info->nInputSize;
    m_output_num = m_io_info->nOutputSize;

    // Build name-to-index maps
    m_input_name_to_idx.clear();
    for (int i = 0; i < m_input_num; ++i) {
        m_input_name_to_idx[std::string(m_io_info->pInputs[i].pName)] = i;
    }
    m_output_name_to_idx.clear();
    for (int i = 0; i < m_output_num; ++i) {
        m_output_name_to_idx[std::string(m_io_info->pOutputs[i].pName)] = i;
    }

    // 5. Prepare IO buffers
    ret = utils::prepare_io("enc", m_io_info, m_io, utils::IO_BUFFER_STRATEGY_DEFAULT);
    if (0 != ret) {
        printf("prepare io failed!\n");
        utils::free_io(m_io);
        return deinit_handle();
    }

    m_handle = handle;
    m_hasInit = true;
    return 0;
}

int EngineWrapper::SetInput(void* pInput, int index) {
    if (!m_hasInit || index < 0 || index >= m_input_num) return -1;
    return utils::push_io_input(pInput, index, m_io);
}

int EngineWrapper::RunSync() {
    if (!m_hasInit) return -1;
    auto ret = AX_ENGINE_RunSync(m_handle, &m_io);
    if (0 != ret) {
        printf("AX_ENGINE_RunSync failed. ret=0x%x\n", ret);
    }
    return ret;
}

int EngineWrapper::GetOutput(void* pOutput, int index) {
    if (!m_hasInit || index < 0 || index >= m_output_num) return -1;
    return utils::push_io_output(pOutput, index, m_io);
}

int EngineWrapper::SetInputByName(const char* name, void* pInput) {
    auto it = m_input_name_to_idx.find(std::string(name));
    if (it == m_input_name_to_idx.end()) {
        printf("Input '%s' not found in model\n", name);
        return -1;
    }
    return SetInput(pInput, it->second);
}

int EngineWrapper::GetOutputByName(const char* name, void* pOutput) {
    auto it = m_output_name_to_idx.find(std::string(name));
    if (it == m_output_name_to_idx.end()) {
        printf("Output '%s' not found in model\n", name);
        return -1;
    }
    return GetOutput(pOutput, it->second);
}

int EngineWrapper::GetInputSize(int index) {
    if (index < 0 || index >= m_input_num) return -1;
    return m_io.pInputs[index].nSize;
}

int EngineWrapper::GetOutputSize(int index) {
    if (index < 0 || index >= m_output_num) return -1;
    return m_io.pOutputs[index].nSize;
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

const char* EngineWrapper::GetInputName(int index) const {
    if (index < 0 || index >= m_input_num) return nullptr;
    return m_io_info->pInputs[index].pName;
}

const char* EngineWrapper::GetOutputName(int index) const {
    if (index < 0 || index >= m_output_num) return nullptr;
    return m_io_info->pOutputs[index].pName;
}

static const char* dtype_str(AX_ENGINE_DATA_TYPE_T t) {
    switch (t) {
        case AX_ENGINE_DT_FLOAT32: return "float32";
        case AX_ENGINE_DT_FLOAT64: return "float64";
        case AX_ENGINE_DT_SINT8:   return "sint8";
        case AX_ENGINE_DT_UINT8:   return "uint8";
        case AX_ENGINE_DT_SINT16:  return "sint16";
        case AX_ENGINE_DT_UINT16:  return "uint16";
        case AX_ENGINE_DT_SINT32:  return "sint32";
        case AX_ENGINE_DT_UINT32:  return "uint32";
        default: return "unknown";
    }
}

const char* EngineWrapper::GetInputDtypeStr(int index) const {
    if (index < 0 || index >= m_input_num) return "?";
    return dtype_str(m_io_info->pInputs[index].eDataType);
}

const char* EngineWrapper::GetOutputDtypeStr(int index) const {
    if (index < 0 || index >= m_output_num) return "?";
    return dtype_str(m_io_info->pOutputs[index].eDataType);
}

int EngineWrapper::Release() {
    if (m_handle) {
        utils::free_io(m_io);
        AX_ENGINE_DestroyHandle(m_handle);
        m_handle = nullptr;
    }
    m_hasInit = false;
    m_input_name_to_idx.clear();
    m_output_name_to_idx.clear();
    return 0;
}
