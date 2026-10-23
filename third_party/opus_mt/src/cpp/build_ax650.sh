#!/bin/bash
# Cross-compile opus_mt_ax650 for AX650N (aarch64).
#
#   TOOLCHAIN_ROOT=/path/to/gcc-arm-...-aarch64-none-linux-gnu \
#   BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
#   bash build_ax650.sh
#
# The script also builds the bundled sentencepiece 0.2.1 static library for
# aarch64 the first time it runs (third_party/sentencepiece_build/).
set -euo pipefail
cd "$(dirname "$0")"

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$PWD/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
BSP_MSP_DIR="${BSP_MSP_DIR:-$PWD/toolchains/ax650n_bsp_sdk/msp/out}"
CC="$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc"
CXX="$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-g++"

for f in "$CC" "$CXX"; do
    [ -x "$f" ] || { echo "missing cross compiler: $f"; echo "run: bash download_toolchains.sh"; exit 1; }
done
[ -d "$BSP_MSP_DIR/include" ] || { echo "missing BSP: $BSP_MSP_DIR"; echo "run: bash download_toolchains.sh"; exit 1; }

SPM_SRC="$PWD/third_party/sentencepiece"
SPM_BUILD="$PWD/third_party/sentencepiece_build"
if [ ! -f "$SPM_SRC/sentencepiece/src/sentencepiece_processor.h" ]; then
    echo "==> extracting bundled sentencepiece source"
    rm -rf "$SPM_SRC" && mkdir -p "$SPM_SRC"
    tar xzf third_party/sentencepiece-0.2.1.tar.gz -C "$SPM_SRC" --strip-components=1
fi
if [ ! -f "$SPM_BUILD/src/libsentencepiece.a" ]; then
    echo "==> cross-building sentencepiece for aarch64"
    cmake -S "$SPM_SRC/sentencepiece" -B "$SPM_BUILD" \
        -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TCMALLOC=OFF -DSPM_BUILD_TEST=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SPM_BUILD" -j"$(nproc)"
fi

cmake -S . -B build_ax650 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DBSP_MSP_DIR="$BSP_MSP_DIR" \
    -DSPM_ROOT="$SPM_SRC/sentencepiece/src" \
    -DSPM_LIB="$SPM_BUILD/src/libsentencepiece.a" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build_ax650 -j"$(nproc)"

mkdir -p bin && cp build_ax650/opus_mt_ax650 bin/
echo "built: $PWD/bin/opus_mt_ax650"
