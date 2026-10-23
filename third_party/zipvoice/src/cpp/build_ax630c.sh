#!/bin/bash
# Cross-compile ZipVoice for AX630C (AXERA demo board)
#
# Prerequisites:
#   - AX630C BSP SDK: /data/shared/huyuan/toolchains/ax620e_bsp_sdk/ (run download_bsp.sh first)
#   - Cross compiler: /data/shared/huyuan/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build_ax630c
mkdir -p build_ax630c && cd build_ax630c

cmake .. \
  -DAX630C=ON \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=../install/ax630c \
  -DCMAKE_BUILD_TYPE=Release \
  $@

make -j$(nproc)
make install

echo ""
echo "========================================"
echo "Build complete!"
echo "Executable: $(pwd)/../install/ax630c/zipvoice"
echo "========================================"
