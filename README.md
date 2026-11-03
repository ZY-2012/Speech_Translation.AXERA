# Speech-Translation-ZipVoice

AX650N 板端实时语音翻译：**连续流式 SenseVoice + 离线定稿 → Opus-MT / Qwen3-0.6B → ZipVoice Distill（语音克隆）**。

这个工程与旧的 `Streaming-Speech-Translation` 完全独立：

- ASR 使用**连续流式 SenseVoice**，轮次结束再用离线模型定稿，避免 1 s 碎片丢上下文；
- TTS 使用 **ZipVoice.AXERA distill C++ 常驻 worker**，不再使用 Pocket-TTS；
- 原文/译文合并成一个对话流，一句原文一句译文；
- 模型全部在主机/NFS，不下载到板端本地盘。

## 目录

```
app/
  asr_stream.py       # 常驻 stream-asr-worker：连续流式 SenseVoice
  asr_offline.py      # AX_ASR_RunPCM：轮次离线定稿 / 回退
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

页面上可以运行时切换 ASR / MT / TTS：

- **ASR**：`流式 + 离线定稿` / `仅离线定稿`；
- **MT**：`Opus-MT 快速` / `Qwen 质量` / `关闭翻译`；
- **TTS**：`ZipVoice 克隆` / `关闭`；播放方式默认 `低延迟（小句合成）`，可选 `连贯优先（整句合成）`；参考音色 `自动` / `英文` / `中文`。

自定义参考音：

- 页面“自定义音色”可上传 3–5 秒 WAV（16/24/44.1 kHz 均可，单/双声道取第一声道）；
- 必须同时填写该段音频的准确文字（1–120 字），否则克隆效果会变差；
- 上传后服务端启动第三个 ZipVoice worker，约 20 秒加载完成；
- 上传期间不能开始会话；重启服务后需要重新上传。

`MT` 细节：

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

## 数字读法

ZipVoice 的 tokenizer 虽然接受数字字符，但声学模型经常直接跳过。因此在送 TTS/显示前会先把阿拉伯数字规范化：

- `50` → 中文 `五十`，英文 `fifty`
- `3.14` → `三点一四` / `three point one four`
- `50%` → `百分之五十` / `fifty percent`
- 带单位的长数字按数量读：`100000000元` → `一亿元`
- **10 位以上且没有单位/量词**的连续数字，按位读：`13800138000` → `一三八零零一三八零零零` / `one three eight zero zero one three eight zero zero zero`
- 带前导零的编号按位读：`007` → `零零七` / `zero zero seven`
- 中文年份按位读：`2026年` → `二零二六年`

实现见 `app/number_text.py`；原文和译文都会规范化后再送 MT/TTS。

## 关键参数

`config/board.yaml`：

- `audio.min_speech_ms: 500`：短于 0.5 s 的噪声段直接丢弃；
- `audio.min_rms: 0.003`：低于该 RMS 的近静音段不送 ASR，抑制 SenseVoice 在静音上的整句幻觉；
- `audio.pre_roll_ms: 800` / `audio.post_roll_ms: 600`：留少量上下文，避免把长段环境噪声送进 ASR；
- `audio.trailing_silence_ms: 1000`：闭句等待，减少一句话被切成碎片；
- `audio.sentence_merge_ms: 0`：流式 ASR 自己保留上下文，不再做文本/音频拼接；
- `audio.max_segment_ms: 10000`：最长轮次，超过强制成段；
- `vad.backend: ten`：TEN-VAD，替代能量 VAD；能量 VAD 在自然停顿处会把 70s 录音切成 31 个 1s 碎片，TEN-VAD 只切 19 个 4～6s 段；
- `asr.stream_enabled: true`：连续 streaming SenseVoice + 离线定稿；`asr.model_dir` 仍用于定稿/回退；
- `asr.stream_model_dir: models/asr_chain/asr`：流式 SenseVoice 模型；
- `asr.min_text_chars: 2` / `asr.drop_unpunctuated_chars: 12`：过滤纯标点、单字噪声，以及没有句末标点的短中文碎片；
- `llm.backend: opus|qwen|none`：默认 Opus，Qwen 保留为质量模式，`none` 只输出原文；
- `llm.opus.model_root`：`models/opus_mt/models`，包含 `zh-en` 和 `en-zh`；
- `tts.engine: zipvoice`：ZipVoice distill；
- `tts.speed: 0.9`：语速，数值越小越慢；
- `tts.voice: auto|en|zh`：参考音色。`auto` 按目标语言选英文/中文参考音；`en`/`zh` 强制使用对应参考音；
- `tts.zh_prompt_wav/zh_prompt_text`：EN→ZH 使用的中文参考音和转写文本，默认 `zh_1_4p5s.wav`；
- `tts.continuous: false`：默认按目标小句合成，首音低延迟；设为 `true` 时整句一次提交，播放更连贯但语速/停顿可能不均匀；
- `tts.prompt_wav/prompt_text`：默认英文克隆参考音；
- `tts.clause_max_chars: 40`：目标小句长度，控制首帧等待。

## 已验证

- 连续流式 SenseVoice partial 约 0.5 s 开始出字；
- 离线 SenseVoice：6.6 s 音频约 0.21 s，RTF ≈ 0.03，输出带中文标点；
- TEN-VAD + 离线 SenseVoice：`vad_example` 前 10s 从 31 个 1s 碎片变为 4.6s/3.3s 两个完整段，识别从“试错的过程很简”恢复为“试错的过程很简单而且特别是今天报名唱缺卡的同学，你们可以”；
- ZipVoice distill C++：短句约 RTF 0.7（首句有固定 decoder 开销），模型常驻后不再重载；
- 整句 TTS（可选）：6.4 s 英文音频约 1.6 s 合成完成，播放端无句间空档；首音约在闭句后 2.1 s；
- 完整 WSS：流式 partial、离线定稿、Opus-MT 翻译、ZipVoice PCM 回传正常；
- Opus-MT AX650 C++（50 板）：zh→en 约 57 ms，en→zh 约 108 ms（含首句），BLEU 32.78 / 28.72；
- 0.5 s 以下语音段被丢弃；纯标点/单字噪声结果不再建对话对，也不再触发 MT/TTS；
- 已改为 **连续流式 SenseVoice + 轮次离线定稿**：VAD 只决定 flush 时点，ASR 不再按 1 s 片段重建上下文；
- 页面不再显示延迟指标和运行日志；每句 meta 只保留编号、语音时长和目标语言。

## 限制

- 流式 partial 先出字，最终译文使用轮次离线定稿；不是逐字字幕；
- `models/asr_chain/vad/ten-vad/ten-vad.axmodel` 和 `third_party/ten_vad/libten_vad.so` 是 TEN-VAD 依赖；
- ZipVoice 默认按小句合成（低延迟），首帧约 1.4 s；相邻小句之间可能有合成空档。整句模式更连贯，但首音更晚且语速可能不均匀；
- 默认 `tts.voice: auto` 会为中文→英文使用英文参考音、英文→中文使用中文参考音；两个 ZipVoice worker 常驻，避免 EN→ZH 误用英文音色；
- Opus-MT 是固定语言对模型：`zh-en` 和 `en-zh` 各一个，不支持第三种语言；
- 单麦克风无法可靠区分“本人”和旁边说话的人；如果旁边人声与本人音量接近，建议使用指向性/耳麦麦克风，或在系统层暂停输入；
- 流式 partial 只在 VAD 已累计约 600 ms 真实语音后显示；纯“嗯/啊”等填充词、单字和短碎片不会进入识别卡片或触发 MT/TTS；
- Qwen 和 Opus 同时常驻会占更多内存；50 板启动约需 2～3 分钟；
- 译文音频时长可能长于源语音，连续长时说话仍需要变速或更简洁的译文来控制积压；
- **页面麦克风默认开启防回声**：译音播放期间麦克风和 ASR 输入都会被静音，避免扬声器声音被再次识别成 `The.` / `Ran.` / `Yeah.`。WAV 模拟会自动关闭防回声；戴耳机时也可以取消勾选，恢复边说边译；
- ZipVoice 的 `num_step=3` / `speed=1.1` 更快配置实测会丢词（`The process of trial.` 只读出 `The process.`），所以默认保留 **4 步 + speed 1.0**；首帧约 1.4 s。
