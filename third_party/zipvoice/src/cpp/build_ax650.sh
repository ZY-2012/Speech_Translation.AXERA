#!/bin/bash
# Cross-compile ZipVoice for AXERA (AX650 demo board)
#
# Prerequisites:
#   - AX650 BSP SDK: cpp/ax650n_bsp_sdk/  (run download_bsp.sh first)
#   - Cross compiler: /data/shared/huyuan/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build_ax650
mkdir -p build_ax650 && cd build_ax650

cmake .. \
  -DAX650=ON \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=../install/ax650 \
  -DCMAKE_BUILD_TYPE=Release \
  $@

make -j$(nproc)
make install

echo ""
echo "========================================"
echo "Build complete!"
echo "Executable: $(pwd)/../install/ax650/zipvoice"
echo "========================================"
