#!/bin/bash
# Download BSP SDKs to /data/shared/huyuan/toolchains/

TARGET=/data/shared/huyuan/toolchains
mkdir -p $TARGET && cd $TARGET

if [ ! -d ax650n_bsp_sdk ]; then
  echo "Downloading ax650n_bsp_sdk..."
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
fi

if [ ! -d ax620e_bsp_sdk ]; then
  echo "Downloading ax620e_bsp_sdk..."
  git clone https://github.com/AXERA-TECH/ax620e_bsp_sdk.git --depth=1
fi

if [ ! -d axcl_bsp_sdk ]; then
  echo "Downloading axcl_bsp_sdk..."
  git clone https://github.com/Abandon-ht/axcl_bsp_sdk.git --depth=1
fi

echo "Done. SDKs at $TARGET/"
ls -d ax650n_bsp_sdk ax620e_bsp_sdk axcl_bsp_sdk 2>/dev/null