cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(zipvoice)

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2 -fdata-sections -ffunction-sections")
endif()

# Platform dependencies (AXERA or AXCL)
include(cmake/msp_dependencies.cmake)

include_directories(${MSP_INC_DIR})

# Project includes
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/)

# Core source files (common to AXERA and AXCL)
set(SRC
    src/EngineWrapper.cpp
    src/tokenizer.cpp
    src/fbank.cpp
    src/zipvoice_engine.cpp
    src/vocoder.cpp
    third_party/kissfft/kiss_fft.c
)

# AXCL-only source files (middleware + utilities)
if(AXCL)
    aux_source_directory(${CMAKE_SOURCE_DIR}/src/middleware AXCL_SRCS)
    aux_source_directory(${CMAKE_SOURCE_DIR}/src/utilities  AXCL_SRCS)
    list(APPEND SRC ${AXCL_SRCS})
    message(STATUS "AXCL middleware: ${AXCL_SRCS}")
endif()

add_executable(${PROJECT_NAME} zipvoice.cpp ${SRC})
target_include_directories(${PROJECT_NAME} PRIVATE third_party/kissfft)
target_link_libraries(${PROJECT_NAME} ${MSP_LIBS})

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ./)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/")

add_executable(zipvoice_worker zipvoice_worker.cpp ${SRC})
target_include_directories(zipvoice_worker PRIVATE third_party/kissfft)
target_link_libraries(zipvoice_worker ${MSP_LIBS})
install(TARGETS zipvoice_worker
        RUNTIME DESTINATION ./)
set_target_properties(zipvoice_worker
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/")

message(STATUS "========================================")
message(STATUS "ZipVoice C++ Build Configuration")
message(STATUS "========================================")
