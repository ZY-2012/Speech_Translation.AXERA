#!/bin/bash
# Cross-compile ZipVoice for AXCL (AXCL compute card)
#
# Prerequisites:
#   - AXCL BSP SDK: /data/shared/huyuan/toolchains/axcl_bsp_sdk/
#   - Cross compiler: /data/shared/huyuan/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/
#
# Clone SDK (one-time):
#   cd /data/shared/huyuan/toolchains
#   git clone https://github.com/Abandon-ht/axcl_bsp_sdk.git --depth=1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build_axcl
mkdir -p build_axcl && cd build_axcl

cmake .. \
  -DAXCL=ON \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=../install/axcl \
  -DCMAKE_BUILD_TYPE=Release \
  $@

make -j$(nproc)
make install

echo ""
echo "========================================"
echo "Build complete!"
echo "Executable: $(pwd)/../install/axcl/zipvoice"
echo "========================================"
