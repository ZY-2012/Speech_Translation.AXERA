# Speech-Translation-ZipVoice

AX650N 板端实时语音翻译：**离线 SenseVoice（或流式 Zipformer）→ FireRedPunc → Opus-MT / Qwen3-0.6B → ZipVoice Distill（语音克隆）**。

这个工程与旧的 `Streaming-Speech-Translation` 完全独立：

- ASR 改为**非流式小句识别**，避免 0.2 s 短段产生 `i뭐the` 这类乱码；
- TTS 使用 **ZipVoice.AXERA distill C++ 常驻 worker**，不再使用 Pocket-TTS；
- 原文/译文合并成一个对话流，一句原文一句译文；
- 模型全部在主机/NFS，不下载到板端本地盘。

## 目录

```
app/
  asr_stream.py       # 常驻 stream-asr-worker：连续流式 SenseVoice / Zipformer
  asr_offline.py      # AX_ASR_RunPCM：轮次离线定稿 / 回退
  punc.py             # FireRedPunc AX650 常驻 worker 客户端（auto 模式）
  punc_worker.py      # FireRedPunc worker（系统 Python 3.10）
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
  zf/zf_asr_worker.py # Zipformer 流式 ASR worker（可选后端）
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
# 50 板（SenseVoice + Opus 默认）
CONFIG=config/board50.yaml PORT=8097 TLS_PORT=8445 bash scripts/run_board.sh
# 50 板 Zipformer 流式 + Punc A/B（Opus，Qwen 关闭）
CONFIG=config/board50_zf.yaml PORT=8098 bash scripts/run_board.sh
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

首轮启动会加载离线 ASR、Opus-MT/Qwen、ZipVoice 和 FireRedPunc；模型缓存后会明显变快。
`config/board50_zf.yaml` 使用 Zipformer 流式 ASR + Punc，端口 8098，Qwen 关闭。

`config/tls/` 是板端自签名证书，不提交到仓库。没有证书时只开 HTTP；需要麦克风时生成：

```bash
openssl req -x509 -newkey rsa:2048 -nodes   -keyout config/tls/server.key -out config/tls/server.crt -days 3650   -subj "/CN=10.126.29.50"   -addext "subjectAltName=IP:10.126.29.50,IP:10.126.29.171,DNS:localhost"
```

## FireRedPunc 标点模型

Punc 工程在 `../FireRedPunc.AXERA`（与本工程同级），模型 `models/punc.axmodel`
在 AX650 NPU1 上约 23 ms/128 token。`config/board50.yaml` 默认 `punc.mode: auto`：
SenseVoice 文本已有 `。！？!?` 时直接跳过，不改变现有稳定链路；Zipformer
文本无标点时自动补标点，再走 `split_source -> MT -> TTS`。

```yaml
punc:
  enabled: true
  mode: auto
  python: /usr/bin/python3
  worker: app/punc_worker.py
  runtime_dir: ../FireRedPunc.AXERA/python
  model_dir: ../FireRedPunc.AXERA/models
```

`app/punc_worker.py` 用系统 Python 3.10（有 transformers + axengine），主服务
仍用 portable Python 3.11；worker 会清掉主进程的 `PYTHONPATH`，避免 3.11 的
numpy 泄漏到 3.10。

## Zipformer 可选 ASR 后端

`config/board50_zf.yaml` 把 ASR 换成 ZF 流式模型：

```yaml
asr:
  backend: zipformer
  stream_worker: third_party/zf/zf_asr_worker.py
  stream_model_dir: ../../ZF/streaming/release/models
```

`third_party/zf/zf_asr_worker.py` 用系统 Python 3.10 + `kaldi_native_fbank` +
`axengine`，维护 35 个 encoder 状态，按 32 帧 chunk 增量解码，VAD 只决定 flush。
模型在 `../ZF/streaming/release/models`（AX650 U16）。

A/B 观察（50 板，`samples/zh_6s.wav` / `samples/en_16k.wav`）：

- ZF 增量 ASR 在 VAD 结束后的 final 延迟约 50–78 ms（含 Punc 24–35 ms）；
- 首帧 TTS：`num_step=4` 约 1.33–1.39 s，`num_step=3` 约 1.08–1.15 s；
  `config/board50_zf.yaml` 采用 3 步 + `trailing_silence_ms: 700`，
  “闭句→首帧音”从约 1.5–2.1 s 降到约 1.2–1.3 s；
- ZipVoice Python 分词器已缓存固定 prompt 的 token ids，后续每句只分词新译文，
  分词耗时从约 505 ms 降到约 260–270 ms；
- SenseVoice 当前离线定稿仍是中文/英文质量更稳的那一路；
- ZF 在这两条样本上出现 `莫名昌卡` / `fifty pieces of cold` 一类识别错误，
  因此默认仍是 SenseVoice，ZF 需要更多域内语料再决定是否切换。

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
- `audio.pre_roll_ms: 2000` / `audio.post_roll_ms: 600`：给 VAD 边界留足上下文，减少句首/句尾吞字；
- `audio.trailing_silence_ms: 1000`：闭句等待，减少一句话被切成碎片；
  `config/board50_zf.yaml` 为提速改为 **700**（`zh_30s.wav` 上切段结果与 1000 相同）；
- `audio.sentence_merge_ms: 300`：相邻 VAD 段的 PCM 在 0.3 s 内继续出现时按**音频级**合并；优先速度，长停顿仍按新句处理；
- `audio.max_segment_ms: 6000`：最长小句，避免等整段长语音；
- `vad.backend: ten`：TEN-VAD，替代能量 VAD；能量 VAD 在自然停顿处会把 70s 录音切成 31 个 1s 碎片，TEN-VAD 只切 19 个 4～6s 段；
- `asr.stream_enabled: true`：连续 streaming SenseVoice + 离线定稿；`asr.model_dir` 仍用于定稿/回退；
- `asr.stream_model_dir: models/asr_chain/asr`：流式 SenseVoice 模型；
- `asr.backend: sensevoice|zipformer`：`zipformer` 时改用 `config/board50_zf.yaml`；
- `punc.mode: auto|always|off`：`auto` 只对无句末标点的文本运行 Punc；
- `llm.backend: opus|qwen`：默认 Opus，Qwen 保留为质量模式；
- `llm.opus.model_root`：`models/opus_mt/models`，包含 `zh-en` 和 `en-zh`；
- `tts.engine: zipvoice`：ZipVoice distill；
- `tts.prompt_wav/prompt_text`：默认英文克隆参考音；
- `tts.clause_max_chars: 40`：目标小句长度，控制首帧等待；
- `tts.num_step: 4`：默认质量档；`config/board50_zf.yaml` 用 **3** 提速，
  5 条英文样本经 SenseVoice 回识与 4 步一致；若某域出现丢词，改回 4；
- ZipVoice 模型包里的 `models/zipvoice/cpp/scripts/py_daemon.py` 需要与
  `third_party/zipvoice/src/cpp/scripts/py_daemon.py` 同步，否则没有 prompt 缓存。

## 已验证

- 离线 SenseVoice：6.6 s 音频约 0.21 s，RTF ≈ 0.03，输出带中文标点；
- TEN-VAD + 离线 SenseVoice：`vad_example` 前 10s 从 31 个 1s 碎片变为 4.6s/3.3s 两个完整段，识别从“试错的过程很简”恢复为“试错的过程很简单而且特别是今天报名唱缺卡的同学，你们可以”；
- ZipVoice distill C++：短句约 RTF 0.7（首句有固定 decoder 开销），模型常驻后不再重载；
- 完整 WSS：离线 ASR 定稿、Opus-MT 翻译、ZipVoice PCM 回传正常；
- FireRedPunc AX650：板端 token/text 与 fp32 100% 一致，128 token 约 23 ms；
  SenseVoice 文本自动跳过，Zipformer 文本自动补标点；
- Zipformer 流式 + Punc：`samples/zh_6s.wav` / `en_16k.wav` / ZF `0.wav`
  全链路跑通，VAD 结束后 final 约 50–62 ms；
- Opus-MT AX650 C++（50 板）：zh→en 约 57 ms，en→zh 约 108 ms（含首句），BLEU 32.78 / 28.72；
- 0.6 s 以下语音段被丢弃，不再产生短段乱码；
- 流式 partial 在 0.5 s 内开始出字；轮次结束用离线模型补标点/纠错；
- 已改为 **连续流式 SenseVoice + 轮次离线定稿**：VAD 只决定 flush 时点，ASR 不再按 1 s 片段重建上下文；
- 页面上的“识别”指标只显示真实 ASR 耗时；`合并等待` 是 utterance buffer 的端点等待，会单独显示在 meta 里。

## 限制

- 非流式 ASR 在段末才出字，适合“小句级同时传译”，不是逐字字幕；
- `models/asr_chain/vad/ten-vad/ten-vad.axmodel` 和 `third_party/ten_vad/libten_vad.so` 是 TEN-VAD 依赖；
- ZipVoice 是整小句合成后输出 PCM；3 步/700 ms 配置下首帧约 1.1 s，
  4 步约 1.3–1.4 s；不是逐音素流式；
- 当前默认英文克隆参考音，适合中文→英文；反向翻译可替换 `prompt_wav/prompt_text`；
- Opus-MT 是固定语言对模型：`zh-en` 和 `en-zh` 各一个，不支持第三种语言；
- Qwen 和 Opus 同时常驻会占更多内存；50 板启动约需 2～3 分钟；NFS 冷缓存时
  首次加载可能到 8～12 分钟，`run_board.sh` 已把等待上限放宽到 900 s；
- Punc 依赖同级的 `../FireRedPunc.AXERA`；目录不存在时 Punc worker 启动失败，
  主链路会自动跳过 Punc，不影响 SenseVoice 翻译；
- FireRedPunc 会把英文转成小写（上游 `RuleBaedTxtFix` 行为）；en→zh 对大小写
  敏感时可再按原始字符恢复；
- Zipformer 当前只作为 `config/board50_zf.yaml` 的可选后端，默认仍是 SenseVoice；
- 译文音频时长可能长于源语音，连续长时说话仍需要变速或更简洁的译文来控制积压；
- **防回声默认开启（页面 + 服务端双重）**：译音播放期间麦克风和 ASR 输入都会被静音，避免扬声器声音被再次识别成 `The.` / `Ran.` / `Yeah.`。戴耳机时可以取消该勾选，恢复边说边译；
- `num_step=3` + `speed=1.1` 的旧配置实测会丢词；当前 ZF 配置是
  **3 步 + speed 1.0**，5 条英文样本回识完整。若换域后丢词，改回 4 步；
  默认 `board50.yaml` 保持 4 步。
