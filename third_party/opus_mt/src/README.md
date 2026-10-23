# Opus-MT.AXERA

Helsinki-NLP OPUS-MT 中英互译模型（Marian encoder-decoder）的 Axera AX650N 部署工程：
中文 → 英文、英文 → 中文，支持 X86 导出 + Pulsar2 U16 量化 + Python/C++ 板端推理。

- [x] 静态 ONNX 导出 + PyTorch/ONNX 对分
- [x] 真实数据校准集 + Pulsar2 U16/S8 量化
- [x] Python 板端推理（`python/`，纯 numpy + sentencepiece + axengine）
- [x] C++ 板端推理（`cpp/`，交叉编译，内置 sentencepiece）
- [x] 200 句 Tatoeba 评测 + 板端 C++/Python 逐句一致

量化模型与预编译二进制在 HuggingFace：
[HY-2012/Opus-MT.AXERA](https://huggingface.co/HY-2012/Opus-MT.AXERA)

上游模型：
[Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) ·
[Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)

## 结果（AX650N，batch=1，已预热）

Tatoeba test 200 句，greedy 解码：

| 方向 | 路径 | BLEU | chrF | 与 FP32 一致 | 延迟 |
|------|------|------|------|--------------|------|
| zh→en | PyTorch FP32 | 33.48 | 54.73 | — | — |
| zh→en | **AX650 Python** | 32.78 | 53.91 | 179/200 | 69.1 ms/句 |
| zh→en | **AX650 C++** | 32.78 | 53.91 | 200/200 vs Python | **58.5 ms/句** |
| en→zh | PyTorch FP32 | 28.63 | 24.49 | — | — |
| en→zh | **AX650 Python** | 28.72 | 24.65 | 168/200 | 56.2 ms/句 |
| en→zh | **AX650 C++** | 28.72 | 24.65 | 200/200 vs Python | **47.3 ms/句** |

- C++ 与 Python 输出 200/200 完全相同（两个方向）
- encoder 4.90 ms，decoder **5.41 ms/step**（C++）；Python 为 5.5 / 6.4 ms
- 逐张量 cosine（20 组）：encoder 均值 0.9995~0.9998，decoder logits 均值 0.9972~0.9996
- 模型：encoder 83 MiB + decoder 124 MiB（U16 激活 / S8 权重），方向间共享同一套代码

## 支持平台

| 平台 | Python | C++ | 状态 |
|------|--------|-----|------|
| AX650N | ✅ | ✅ | 板端实测 |
| AX630C / AXCL | — | — | 未验证（Pulsar2 重新编译后可复用同一套源码） |

## 模型接口

| 图 | 输入 | 输出 |
|----|------|------|
| encoder | `input_ids[1,128] int32`, `attention_mask[1,128] fp32` | `encoder_hidden_states[1,128,512] fp32` |
| decoder | `decoder_input_ids[1,64] int32`, `decoder_attention_mask[1,64] int32`, `position_ids[1,64] int32`, `encoder_hidden_states[1,128,512] fp32`, `encoder_attention_mask[1,128] fp32` | `logits[1,1,65001] fp32` |

decoder 采用**左对齐前缀 + 显式 position_ids**：每步把当前前缀右对齐到 64 槽，
最后一槽就是当前 token，图内只对最后一个位置做 LM head。这样既避免了 Pulsar2 对
动态 `Gather` 的不支持，也把每步输出从 `[1,64,65001]` 降到 `[1,1,65001]`。

## 目录结构

```
Opus-MT.AXERA/
├── model_convert/       # ONNX 导出 → 对分 → 校准数据 → Pulsar2 U16 量化
├── python/              # Python 板端推理 SDK（axengine）
├── cpp/                 # C++ 板端推理源码 + 交叉编译脚本
├── models/              # 仅 README，量化模型从 HuggingFace 下载
├── samples/             # 中英文示例输入
└── requirements.txt
```

## 环境

```bash
# 推理依赖（板端）
pip install -r requirements.txt

# 转换依赖（X86）
pip install -r model_convert/requirements.txt
```

Pulsar2 量化环境见 [model_convert/README.md](model_convert/README.md)。

## 快速开始（板端推理）

```bash
# 1) 从 HuggingFace 取量化模型
hf download HY-2012/Opus-MT.AXERA --local-dir ./_hf_download
cp -r ./_hf_download/models .

# 2) Python
python python/translate.py --direction zh-en --model-root models --text "这是一个测试。"

# 3) C++
bash cpp/download_toolchains.sh        # gcc 交叉编译器 + AX650N BSP
bash cpp/build_ax650.sh                # -> cpp/bin/opus_mt_ax650
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./cpp/bin/opus_mt_ax650 --direction zh-en --model-root models --text "这是一个测试。"
```

`en-zh` 会自动补 `>>cmn_Hans<<` 前缀；如需其他中文变体可自行传入
`>>yue_Hant<<` 等前缀。批量输入用 `--input file --output file`，每行一句。

## 模型转换

完整步骤见 [model_convert/README.md](model_convert/README.md)：

```bash
export OPUS_MT_ROOT=$PWD

# 1) 下载浮点模型到 origin/zh-en、origin/en-zh
#    hf download Helsinki-NLP/opus-mt-zh-en --local-dir origin/zh-en
#    hf download Helsinki-NLP/opus-mt-en-zh --local-dir origin/en-zh

# 2) 导出静态 ONNX + 对分（cosine 1.0，贪心 4/4 一致）
python model_convert/export_models.py --direction zh-en
python model_convert/export_models.py --direction en-zh

# 3) 生成真实数据校准集（Tatoeba 200 句样本）
python model_convert/prepare_calib.py --direction zh-en
python model_convert/prepare_calib.py --direction en-zh

# 4) Pulsar2 U16 量化
python model_convert/compile_models.py --direction zh-en --part both --u16
python model_convert/compile_models.py --direction en-zh --part both --u16
```

产物：`zh-en/compile/encoder.axmodel`、`decoder.axmodel`（en-zh 同理）。

### 量化结论

- 纯 INT8 时 decoder logits cosine 只有 0.971，不满足 ≥0.99；改用 **U16 激活 + S8 权重**后
  encoder 0.9998 / decoder 0.9996（zh-en），en-zh 0.9995 / 0.9972
- 校准数据必须是**真实业务数据**：本工程用 Tatoeba 测试集 200 句，encoder 64 句、
  decoder 256 个真实目标前缀（覆盖首步/中段/末段）
- Marian 的 `final_logits_bias` 必须加到 `lm_head` 输出上，否则解码会发散
- decoder 输入使用左对齐前缀 + 显式 `position_ids`，避免动态 Gather 和超大输出

## 致谢 / References

- 上游模型：[Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) ·
  [Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
- OPUS-MT：[Helsinki-NLP/OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT) ·
  [OPUS-MT-train](https://github.com/Helsinki-NLP/OPUS-MT-train) ·
  [Tatoeba-Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge)
- 测试集：[zho-eng](https://object.pouta.csc.fi/Tatoeba-MT-models/zho-eng/opus-2020-07-17.test.txt) ·
  [eng-zho](https://object.pouta.csc.fi/Tatoeba-MT-models/eng-zho/opus-2020-07-17.test.txt) ·
  [Tatoeba](https://tatoeba.org)
- Marian / Transformers：[huggingface/transformers](https://github.com/huggingface/transformers)
- Pulsar2：[文档](https://pulsar2-docs.readthedocs.io) ·
  [发布页](https://huggingface.co/AXERA-TECH/Pulsar2)
- 板端运行时：[AXERA-TECH/pyaxengine](https://github.com/AXERA-TECH/pyaxengine)
- AX650N BSP：[AXERA-TECH/ax650n_bsp_sdk](https://github.com/AXERA-TECH/ax650n_bsp_sdk)
- 工作流参考：[AXERA-TECH/Magnetar](https://github.com/AXERA-TECH/Magnetar)
- SentencePiece：[google/sentencepiece](https://github.com/google/sentencepiece)
- JSON：[nlohmann/json](https://github.com/nlohmann/json)
- 部署工程模板：[ZY-2012/FS-EEND.AXERA](https://github.com/ZY-2012/FS-EEND.AXERA)
- AXERA 开源主页：[github.com/AXERA-TECH](https://github.com/AXERA-TECH)

## License

本工程代码 Apache-2.0。模型权重版权归上游：
`opus-mt-zh-en` 为 CC-BY-4.0，`opus-mt-en-zh` 为 Apache-2.0。
