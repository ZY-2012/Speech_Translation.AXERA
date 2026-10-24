# Speech-Translation-ZipVoice

AX650N 板端实时语音翻译：**离线 SenseVoice → Opus-MT / Qwen3-0.6B → ZipVoice Distill（语音克隆）**。

这个工程与旧的 `Streaming-Speech-Translation` 完全独立：

- ASR 改为**非流式小句识别**，避免 0.2 s 短段产生 `i뭐the` 这类乱码；
- TTS 使用 **ZipVoice.AXERA distill C++ 常驻 worker**，不再使用 Pocket-TTS；
- 原文/译文合并成一个对话流，一句原文一句译文；
- 模型全部在主机/NFS，不下载到板端本地盘。

## 目录

```
app/
  asr_offline.py      # AX_ASR_RunPCM 非流式 SenseVoice
  tts_zipvoice.py     # ZipVoice C++ 常驻 worker 客户端
  pipeline.py         # VAD -> ASR -> 小句 MT -> ZipVoice
  llm_axllm.py        # Qwen3-0.6B axllm OpenAI 流式接口
  opus_mt.py          # Opus-MT AX650 C++ 常驻 worker 客户端
  server.py           # HTTP/HTTPS + WebSocket
static/index.html     # 单对话流 UI
config/board.yaml
models/
  asr_official/...
  qwen3-0.6b/
  opus_mt/            # HY-2012/Opus-MT.AXERA: zh-en + en-zh
    bin/opus_mt_worker
    models/{zh-en,en-zh}/
  zipvoice/
    bin/zipvoice_worker
    models/zipvoice_distill_ax650/
    models/vocoder/vocos_full.axmodel
    resources/zipvoice_hf/zipvoice/tokens.txt
    assets/moss_prompts/
    scripts/          # pypinyin + piper_phonemize tokenizer
    cpp/scripts/py_daemon.py
third_party/
  asr/lib/libax_asr_api.so
  llm/axllm
  opus_mt/src/        # C++ source and opus_mt_worker.cpp
  zipvoice/src/       # C++ source and zipvoice_worker.cpp
runtime/
  python/             # portable aarch64 Python 3.11
  pylibs/             # numpy/scipy/pypinyin/piper_phonemize/...
```

## 板端启动

启动前确认旧的 `Streaming-Speech-Translation` 服务已经停掉，否则两套
ASR/TTS 模型会同时占 NPU。

```bash
cd /root/huyuan/183_workspace/Speech-Translation-ZipVoice
# 50 板（Opus 默认，Qwen 可选）
CONFIG=config/board50.yaml PORT=8097 TLS_PORT=8445 bash scripts/run_board.sh
# 171 板（Qwen 默认，Opus 可选）
bash scripts/run_board.sh
```

端口：

- 50 板：HTTP `8097` / HTTPS `8445`
- 171 板：HTTP `8096` / HTTPS `8444`

页面上的 `MT:` 下拉框可以运行时切换：

- `Opus-MT 快速`：C++ NPU，zh↔en，约 50～110 ms/句；
- `Qwen 质量`：Qwen3-0.6B，约 20 tok/s（50 板约 8 tok/s）。

麦克风必须使用 HTTPS 或 `localhost`。也可以：

```bash
ssh -L 8096:localhost:8096 root@<board-ip>
# 打开 http://localhost:8096/
```

首轮启动会加载离线 ASR、Opus-MT/Qwen 和 ZipVoice；模型缓存后会明显变快。

`config/tls/` 是板端自签名证书，不提交到仓库。没有证书时只开 HTTP；需要麦克风时生成：

```bash
openssl req -x509 -newkey rsa:2048 -nodes   -keyout config/tls/server.key -out config/tls/server.crt -days 3650   -subj "/CN=10.126.29.50"   -addext "subjectAltName=IP:10.126.29.50,IP:10.126.29.171,DNS:localhost"
```

## Opus-MT 模型

```bash
hf download HY-2012/Opus-MT.AXERA --local-dir models/opus_mt
cd third_party/opus_mt/src/cpp
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out bash build_ax650.sh
cp build_ax650/opus_mt_worker ../../../../models/opus_mt/bin/
```

## 关键参数

`config/board.yaml`：

- `audio.min_speech_ms: 500`：短于 0.5 s 的噪声段直接丢弃；
- `audio.trailing_silence_ms: 500`：闭句等待，减少一句话被切成碎片；
- `audio.sentence_merge_ms: 1000`：相邻 VAD 段的 PCM 在 1 s 内继续出现时按**音频级**合并，端点后只跑一次 ASR；
- `audio.max_segment_ms: 4000`：最长小句，避免等整段长语音；
- `asr.backend: offline`：每段调用 `AX_ASR_RunPCM`，输出带标点文本；
- `llm.backend: opus|qwen`：默认 Opus，Qwen 保留为质量模式；
- `llm.opus.model_root`：`models/opus_mt/models`，包含 `zh-en` 和 `en-zh`；
- `tts.engine: zipvoice`：ZipVoice distill；
- `tts.prompt_wav/prompt_text`：默认英文克隆参考音；
- `tts.clause_max_chars: 40`：目标小句长度，控制首帧等待。

## 已验证

- 离线 SenseVoice：6.6 s 音频约 0.21 s，RTF ≈ 0.03，输出带中文标点；
- ZipVoice distill C++：短句约 RTF 0.7（首句有固定 decoder 开销），模型常驻后不再重载；
- 完整 WSS：离线 ASR 定稿、Opus-MT 翻译、ZipVoice PCM 回传正常；
- Opus-MT AX650 C++（50 板）：zh→en 约 57 ms，en→zh 约 108 ms（含首句），BLEU 32.78 / 28.72；
- 0.6 s 以下语音段被丢弃，不再产生短段乱码；
- 已改为音频级 utterance buffer：不再对每个 1 s 片段单独 ASR，也不再做 ASR 后的文本拼接；
- 页面上的“识别”指标只显示真实 ASR 耗时；`合并等待` 是 utterance buffer 的端点等待，会单独显示在 meta 里。

## 限制

- 非流式 ASR 在段末才出字，适合“小句级同时传译”，不是逐字字幕；
- ZipVoice 是整小句合成后输出 PCM，首帧约 1–1.5 s；不是逐音素流式；
- 当前默认英文克隆参考音，适合中文→英文；反向翻译可替换 `prompt_wav/prompt_text`；
- Opus-MT 是固定语言对模型：`zh-en` 和 `en-zh` 各一个，不支持第三种语言；
- Qwen 和 Opus 同时常驻会占更多内存；50 板启动约需 2～3 分钟；
- 译文音频时长可能长于源语音，连续长时说话仍需要变速或更简洁的译文来控制积压；
- **防回声默认开启（页面 + 服务端双重）**：译音播放期间麦克风和 ASR 输入都会被静音，避免扬声器声音被再次识别成 `The.` / `Ran.` / `Yeah.`。戴耳机时可以取消该勾选，恢复边说边译；
- ZipVoice 的 `num_step=3` / `speed=1.1` 更快配置实测会丢词（`The process of trial.` 只读出 `The process.`），所以默认保留 **4 步 + speed 1.0**；首帧约 1.4 s。
