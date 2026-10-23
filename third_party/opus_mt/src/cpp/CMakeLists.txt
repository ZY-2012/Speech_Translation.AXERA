cmake_minimum_required(VERSION 3.15)
project(opus_mt_ax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT BSP_MSP_DIR)
  set(BSP_MSP_DIR "$ENV{BSP_MSP_DIR}")
endif()
if(NOT BSP_MSP_DIR)
  message(FATAL_ERROR "Set BSP_MSP_DIR to <ax650n_bsp_sdk>/msp/out (see cpp/README.md)")
endif()
if(NOT SPM_ROOT)
  message(FATAL_ERROR "Set SPM_ROOT to the extracted sentencepiece source dir")
endif()
if(NOT SPM_LIB)
  message(FATAL_ERROR "Set SPM_LIB to the cross-built libsentencepiece.a")
endif()

include_directories(
  ${BSP_MSP_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${SPM_ROOT}
)
link_directories(${BSP_MSP_DIR}/lib)

add_library(opus_mt_sdk STATIC
  src/ax_engine_runner.cpp
  src/marian_tokenizer.cpp
)
target_link_libraries(opus_mt_sdk PUBLIC
  ${SPM_LIB} ax_engine ax_interpreter ax_sys pthread dl m atomic)

add_executable(opus_mt_ax650 opus_mt_ax.cpp)
target_link_libraries(opus_mt_ax650 PRIVATE opus_mt_sdk)

add_executable(opus_mt_worker opus_mt_worker.cpp)
target_link_libraries(opus_mt_worker PRIVATE opus_mt_sdk)
