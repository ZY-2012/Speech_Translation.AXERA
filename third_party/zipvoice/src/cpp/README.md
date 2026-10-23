# ZipVoice C++ — AXERA & AXCL

## 性能

### Distill 模型 (num_step=4)

| 场景 | 音频 | Python AX650<br>耗时 RTF | C++ AX650<br>耗时 RTF | C++ AXCL<br>耗时 RTF | Python AX630C<br>耗时 RTF | C++ AX630C<br>耗时 RTF |
|------|------|-------------------------|----------------------|---------------------|--------------------------|-----------------------|
| 中文句子 | 6.41s | 1.992s/0.311 | 1.057s/0.165 | 0.918s/**0.143** | 10.296s/1.606 | 4.650s/0.725 |
| 中文段落 | 44.97s | 13.457s/0.301 | 7.357s/0.164 | 6.443s/**0.143** | 71.574s/1.600 | 32.590s/0.725 |
| 英文句子 | 6.41s | 2.045s/0.319 | 1.067s/0.166 | 0.919s/**0.143** | 10.686s/1.667 | 4.654s/0.726 |
| 英文段落 | 59.16s | 19.715s/0.305 | 10.529s/0.178 | 9.207s/**0.156** | 106.183s/1.640 | 46.541s/0.787 |

### 普通模型 (num_step=10)

| 场景 | 音频时长 | Python AX650 (耗时/RTF) | C++ AX650 (耗时/RTF) | C++ AXCL (耗时/RTF) |
|------|---------|------------------------|---------------------|---------------------|
| 中文句子 | 6.41s | 5.781s / 0.902 | 5.709s / 0.891 | 5.452s / **0.850** |
| 中文段落 | 44.97s | 40.292s / 0.901 | 39.714s / 0.883 | 38.138s / **0.848** |
| 英文句子 | 6.41s | 5.711s / 0.891 | 5.693s / 0.888 | 5.447s / **0.850** |
| 英文段落 | 59.16s | 62.161s / 0.960 | 56.759s / 0.957 | 54.451s / **0.920** |

## 环境准备

下载交叉编译器、AXERA SDK、AXCL SDK 到指定目录（记为 `<TOOLCHAIN_DIR>`）：

```bash
bash download_bsp.sh   

# 或手动:
mkdir -p <TOOLCHAIN_DIR> && cd <TOOLCHAIN_DIR>
wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
git clone https://github.com/AXERA-TECH/ax620e_bsp_sdk.git --depth=1
git clone https://github.com/Abandon-ht/axcl_bsp_sdk.git --depth=1
```

修改 `cmake/msp_dependencies.cmake` 和 `toolchains/aarch64-none-linux-gnu.toolchain.cmake` 中的路径。

英文分词需 Python 依赖（运行设备上安装）：
```bash
pip install jieba numpy pypinyin piper_phonemize
```

## 编译

```bash
bash build_ax650.sh    # → install/ax650/zipvoice
bash build_ax630c.sh   # → install/ax630c/zipvoice
bash build_axcl.sh     # → install/axcl/zipvoice
```

## 运行

模型与编译好的可执行文件发布在 HuggingFace 工程
[AXERA-TECH/ZipVoice.AXERA](https://huggingface.co/AXERA-TECH/ZipVoice.AXERA)
（`bin/zipvoice_ax650`、`bin/zipvoice_ax630c`、`bin/zipvoice_axcl`），
下载后在 HUG 工程根目录执行（可执行文件无需编译）：

```bash
# 板端
bash run_ax650.sh distill zh sentence
bash run_ax650.sh distill zh paragraph
bash run_ax650.sh standard zh sentence
bash run_ax650.sh standard zh paragraph
bash run_ax650.sh distill en sentence
bash run_ax650.sh distill en paragraph
bash run_ax650.sh standard en sentence
bash run_ax650.sh standard en paragraph

# 算力卡
bash run_axcl.sh distill zh paragraph
bash run_axcl.sh standard en sentence

# AX630C (仅 distill)
bash run_ax630c.sh zh sentence
bash run_ax630c.sh en paragraph
```

本地自编译时，可执行文件在 `cpp/install/<platform>/zipvoice`，参数见下表。

## 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型目录 | 必填 |
| `--prompt-wav` | 提示音频 | 必填 |
| `--prompt-text` | 提示文本 | 必填 |
| `--text` / `--text-file` | 合成文本 | 二选一 |
| `--vocoder-model` | vocos_full.axmodel 路径 | 必填 |
| `--output-wav` | 输出 WAV | output.wav |
| `--num-step` | 采样步数 | 10 (distill: 4) |
| `--seed` | 随机种子 | 42 |
