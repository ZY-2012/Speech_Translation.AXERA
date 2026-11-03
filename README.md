# Speech-Translation

<p align="center"><img src="Speech_Translation.gif" alt="Speech-Translation demo" width="100%"></p>

AX650N 端侧实时语音翻译：**SenseVoice / Zipformer → FireRedPunc → Opus-MT / Qwen3-0.6B → ZipVoice 语音克隆**。

模型放在主机/NFS，不下载到板端本地盘。

## 功能

- 默认连续流式 SenseVoice + 离线定稿；可选 Zipformer 流式 + FireRedPunc
- MT：Opus-MT / Qwen3-0.6B / 关闭翻译
- TTS：ZipVoice distill 语音克隆，中英文参考音自动匹配
- 支持上传 3–5 秒 WAV + 对应文字作为自定义参考音
- 低延迟小句 / 连贯整句两种播放方式
- 原译文单对话流，HTTP/HTTPS + WebSocket

## 启动

```bash
cd /path/to/Speech-Translation

# 50 板：SenseVoice + Opus（默认）
CONFIG=config/board50.yaml PORT=8097 TLS_PORT=8445 bash scripts/run_board.sh

# 50 板 A/B：Zipformer + Punc（端口 8098，Qwen 关闭）
CONFIG=config/board50_zf.yaml PORT=8098 bash scripts/run_board.sh

# 171 板：Qwen 默认，Opus 可选
bash scripts/run_board.sh
```

| 板子 | HTTP | HTTPS |
| --- | --- | --- |
| 50 板 | 8097 | 8445 |
| 171 板 | 8096 | 8444 |

麦克风需要 HTTPS 或 `localhost`。TLS 证书放在 `config/tls/server.crt`、`config/tls/server.key`。

FireRedPunc 工程和模型放在同级 `../FireRedPunc.AXERA`，Zipformer 模型放在同级 `../ZF/streaming/release/models`；缺少时对应可选链路会跳过。

## 模型

| 模块 | 本地文件 | 来源 |
| --- | --- | --- |
| ASR 离线 | `models/asr_official/sensevoice_ax650/sensevoice.axmodel` | [AXERA-TECH/models-for-asr-chain](https://huggingface.co/AXERA-TECH/models-for-asr-chain) → `asr/sensevoice` |
| ASR 流式 SenseVoice | `models/asr_chain/asr/sensevoice/streaming_sensevoice.axmodel` | [HY-2012/Tmp_Models](https://huggingface.co/HY-2012/Tmp_Models) |
| ASR 流式 Zipformer | `../ZF/streaming/release/models/*.axmodel` | [k2-fsa/icefall](https://github.com/k2-fsa/icefall) |
| TEN-VAD | `models/asr_chain/vad/ten-vad/ten-vad.axmodel` | [models-for-asr-chain](https://huggingface.co/AXERA-TECH/models-for-asr-chain) → `vad/ten-vad` |
| FireRedPunc | `../FireRedPunc.AXERA/models/punc.axmodel` | [FireRedPunc](https://huggingface.co/FireRedTeam/FireRedPunc) |
| Opus-MT ZH↔EN | `models/opus_mt/models/{zh-en,en-zh}/{encoder,decoder}.axmodel` | [HY-2012/Opus-MT.AXERA](https://huggingface.co/HY-2012/Opus-MT.AXERA) · [源码](https://github.com/ZY-2012/Opus-MT.AXERA) |
| Qwen3-0.6B | `models/qwen3-0.6b/qwen3_*.axmodel` | [AXERA-TECH/Qwen3-0.6B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Qwen3-0.6B-GPTQ-Int4) |
| ZipVoice | `models/zipvoice/models/zipvoice_distill_ax650/{encoder,decoder_part0..3}.axmodel` | [AXERA-TECH/ZipVoice.AXERA](https://huggingface.co/AXERA-TECH/ZipVoice.AXERA) · [源码](https://github.com/AXERA-TECH/ZipVoice.AXERA) |
| Vocoder | `models/zipvoice/models/vocoder/vocos_full.axmodel` | [ZipVoice.AXERA](https://huggingface.co/AXERA-TECH/ZipVoice.AXERA) · [vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) |

上游与运行时：

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) · [TEN-VAD](https://github.com/TEN-framework/ten-vad) · [ZipVoice](https://github.com/k2-fsa/ZipVoice)
- [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) · [opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
- [AXERA-TECH/ax-llm](https://github.com/AXERA-TECH/ax-llm)（`third_party/llm/axllm`）

### 下载模型

```bash
pip install -U "huggingface_hub[cli]"

hf download AXERA-TECH/models-for-asr-chain \
  --include "asr/*" "vad/*" --local-dir models/asr_chain

hf download HY-2012/Tmp_Models \
  --include "streaming_sensevoice.axmodel" --local-dir models/asr_stream
cp models/asr_stream/streaming_sensevoice.axmodel models/asr_chain/asr/sensevoice/
mkdir -p models/asr_official/sensevoice_ax650
cp -r models/asr_chain/asr/sensevoice models/asr_official/sensevoice_ax650/

hf download HY-2012/Opus-MT.AXERA --local-dir models/opus_mt
hf download AXERA-TECH/Qwen3-0.6B-GPTQ-Int4 --local-dir models/qwen3-0.6b
hf download AXERA-TECH/ZipVoice.AXERA --local-dir models/zipvoice
```

## 页面

- ASR：流式 + 离线定稿 / 仅离线定稿
- MT：Opus-MT / Qwen / 关闭翻译
- TTS：ZipVoice 克隆 / 关闭；低延迟 / 连贯；参考音色自动 / 英文 / 中文
- 自定义音色：上传 3–5 秒 WAV，填写对应文字，加载完成后可选
