# BSP / SDK Dependencies Configuration
# Supports both AXERA (demo board) and AXCL (compute card)
#
# AXERA: AX650 / AX630C / AX620Q
# AXCL:  AXCL

if (AX650)
    add_definitions(-DAX650)
    if(NOT BSP_MSP_DIR)
        set(BSP_MSP_DIR /data/shared/huyuan/toolchains/ax650n_bsp_sdk/msp/out)
    endif()
    if(NOT EXISTS ${BSP_MSP_DIR})
        message(FATAL_ERROR "BSP_MSP_DIR ${BSP_MSP_DIR} not exist. Run download_bsp.sh first.")
    endif()
    set(MSP_INC_DIR ${BSP_MSP_DIR}/include)
    set(MSP_LIB_DIR ${BSP_MSP_DIR}/lib)
    list(APPEND MSP_LIBS ax_sys ax_engine ax_interpreter)

elseif(AX630C)
    add_definitions(-DAX630C)
    if(NOT BSP_MSP_DIR)
        set(BSP_MSP_DIR /data/shared/huyuan/toolchains/ax620e_bsp_sdk/msp/out/arm64_glibc)
    endif()
    set(MSP_INC_DIR ${BSP_MSP_DIR}/include)
    set(MSP_LIB_DIR ${BSP_MSP_DIR}/lib)
    list(APPEND MSP_LIBS ax_sys ax_engine ax_interpreter)

elseif(AX620Q)
    add_definitions(-DAX620Q)
    if(NOT BSP_MSP_DIR)
        set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/ax620e_bsp_sdk/msp/out/arm_uclibc)
    endif()
    if(NOT EXISTS ${BSP_MSP_DIR})
        message(FATAL_ERROR "BSP_MSP_DIR ${BSP_MSP_DIR} not exist.")
    endif()
    set(MSP_INC_DIR ${BSP_MSP_DIR}/include)
    set(MSP_LIB_DIR ${BSP_MSP_DIR}/lib)
    list(APPEND MSP_LIBS ax_sys ax_engine ax_interpreter)

elseif(AXCL)
    add_definitions(-DAXCL)
    add_definitions(-DENV_AXCL_RUNTIME_API_ENABLE)
    add_definitions(-DENV_AXCL_NATIVE_API_ENABLE)
    add_definitions(-DENV_HAS_STD_FILESYSTEM)
    add_definitions(-DENV_HAS_POSIX_FILE_STAT)

    # AXCL SDK at /data/shared/huyuan/toolchains/axcl_bsp_sdk/out/
    if(NOT AXCL_SDK_DIR)
        set(AXCL_SDK_DIR /data/shared/huyuan/toolchains/axcl_bsp_sdk/out)
    endif()
    if(NOT EXISTS ${AXCL_SDK_DIR})
        message(FATAL_ERROR "AXCL_SDK_DIR ${AXCL_SDK_DIR} not exist. Run download_bsp.sh first.")
    endif()
    set(MSP_INC_DIR ${AXCL_SDK_DIR}/include ${AXCL_SDK_DIR}/bsp)
    set(MSP_LIB_DIR ${AXCL_SDK_DIR}/lib)
    list(APPEND MSP_LIBS
        axcl_rt axcl_pkg axcl_comm axcl_npu spdlog
        axcl_token axcl_pcie_msg axcl_pcie_dma)
    set(CMAKE_CXX_STANDARD 17)

else()
    message(FATAL_ERROR "Unknown platform. Set -DAX650, -DAX630C, -DAX620Q, or -DAXCL.")
endif()

link_directories(${MSP_LIB_DIR})
message(STATUS "MSP_INC_DIR: ${MSP_INC_DIR}")
message(STATUS "MSP_LIB_DIR: ${MSP_LIB_DIR}")
message(STATUS "MSP_LIBS:    ${MSP_LIBS}")
