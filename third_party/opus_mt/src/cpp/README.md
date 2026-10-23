# C++ 板端推理（AX650N）

`opus_mt_ax650`：Marian encoder + decoder 双图，内置 sentencepiece 0.2.1
（静态链接），运行时只依赖 AX650 板端 `/soc/lib` 和 libstdc++。

## 文件

```
cpp/
├── opus_mt_ax.cpp              # CLI：单句 / 批量，greedy 解码
├── include/ax_engine_runner.hpp
├── include/marian_tokenizer.hpp
├── src/ax_engine_runner.cpp    # 命名多 IO 封装 + cache flush + refcount init
├── src/marian_tokenizer.cpp    # sentencepiece + vocab.json + >>lang<< 前缀
├── third_party/nlohmann/json.hpp
├── third_party/sentencepiece-0.2.1.tar.gz
├── build_ax650.sh              # 先交叉编译 sentencepiece，再编译本工程
└── download_toolchains.sh      # gcc 9.2 aarch64 + AX650N BSP
```

## 编译

```bash
# 方式 A：脚本下载工具链到 cpp/toolchains/
bash cpp/download_toolchains.sh
bash cpp/build_ax650.sh

# 方式 B：复用已有工具链
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
bash cpp/build_ax650.sh
```

产物：`cpp/bin/opus_mt_ax650`（aarch64）。首次编译会自动把
`sentencepiece-0.2.1.tar.gz` 解到 `third_party/sentencepiece/`，交叉编译出
`third_party/sentencepiece_build/src/libsentencepiece.a`。两者都在 `.gitignore` 内。

依赖：
- gcc-arm 9.2 aarch64 交叉编译器
- AX650N BSP `msp/out`（`ax_engine_api.h` + `libax_engine.so` / `libax_interpreter.so` / `libax_sys.so`）
- CMake ≥ 3.15
- 链接 `ax_engine ax_interpreter ax_sys pthread dl m atomic`

## 运行

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH

# 单句
./cpp/bin/opus_mt_ax650 --direction zh-en --model-root models \
    --text "这是一个测试。" --stats

# 批量
./cpp/bin/opus_mt_ax650 --direction en-zh --model-root models \
    --input input.txt --output output.txt --stats
```

`--direction en-zh` 且输入不以 `>>` 开头时，自动补 `>>cmn_Hans<< `。

## 性能（AX650N，200 句 Tatoeba，已预热）

| 方向 | encoder | decoder 单步 | 端到端 | Python 端到端 |
|------|---------|--------------|--------|----------------|
| zh→en | 4.90 ms | 5.41 ms | **58.5 ms/句** | 69.1 ms/句 |
| en→zh | 4.89 ms | 5.41 ms | **47.3 ms/句** | 56.2 ms/句 |

C++ 与 Python 的 200 句输出逐句完全一致（两个方向）。

## 实现要点

- **双模型共享 AX_SYS/AX_ENGINE 初始化**：`AxEngineRunner` 内部用引用计数，
  encoder/decoder 两个实例析构后才 deinit。
- **int32 输入**：不能套用常见的 float* 访问器；runner 暴露 `input_data()/input_bytes()`，
  由调用方按 `model_meta.json` 的 dtype `memcpy`。
- **cache 一致性**：所有输入缓冲 `AX_SYS_MemAllocCached`。每次 `AX_ENGINE_RunSync`
  前 flush 输入、后 invalidate 输出；漏掉会随机读到旧数据。
- **decoder 只算最后一个位置**：前缀右对齐到 64 槽，图内 `Slice` 后只过 LM head，
  输出 `[1,1,65001]`。
- **tokenizer**：`>>lang<<` 前缀在 sentencepiece 之前切出；`vocab.json` 是 source/target
  共享词表，解码用 source.spm 的 `DecodePieces`（与 HF MarianTokenizer 一致）。
- **量化模型**：U16 激活 / S8 权重，encoder 83 MiB、decoder 124 MiB。
