# ZipVoice Distill AX630C 量化总结

本文记录当前保留的 ZipVoice distill AX630C 方案。结论先写清楚：当前板端暂用 `fm_decoder_part3` 四段拆分版，8 段拆分和单体/两段/FP32 试验件已清理。

## 1. 当前结论

- 模型：只处理 distill 模型。
- 平台：AX630C。
- Pulsar2 配置：
  - `target_hardware: AX620E`
  - `npu_mode: NPU2`
- 当前有效方案：
  - encoder、decoder part0、part1、part2：使用当前最终 package 中的 axmodel。
  - decoder part3：拆成 4 个 axmodel 串联运行。
- 当前不保留：
  - part3 单体 axmodel。
  - part3 两段版。
  - part3 八段版。
  - HardSoftplus 大范围 FP32 试验版。
  - mosaicfix/reexport/fp32layers/fp32fullhs/fp32hsfix 等临时构建产物。

本轮声音问题定位结果：

- encoder 和 part0 替换成 ONNX 验证时，整体效果与 axmodel 基本一致，说明主要问题不在 encoder/part0。
- part3 替换成 ONNX 后声音明显恢复，说明残余“马赛克/模糊噪声”主要来自 part3 的 AX630C 量化或后端执行误差。
- part3 四段拆分比单体/两段更稳定。
- part3 八段拆分量化 cosine 更高，但主观音质不如四段，因此当前回退并保留四段版。

当前清理后的保留范围：

- 原 AX650 量化入口脚本；最终 axmodel 为生成物，不随 GitHub 保存
- AX630C 最终 ZipVoice distill package 可由脚本重新生成：
  - `model_convert/zipvoice_distill_ax630c`
- AX630C part3 四段复现文件：
  - 四段 ONNX 导出脚本和说明；ONNX 本体为生成物，不随 GitHub 保存
  - 四段校准数据生成脚本；校准数据本体为生成物，不随 GitHub 保存
  - 四段 Pulsar2 config 和构建日志为生成物，不随 GitHub 保存
- vocoder 量化：
  - `model_convert/vocoder_ax630c`
  - `model_convert/vocoder_ax650`
  - `model_convert/vocoder_onnx`
  - 上述均为生成目录，不随 GitHub 保存

vocoder 当前只保留 full axmodel：

```text
model_convert/vocoder_ax650/vocos_full.axmodel
model_convert/vocoder_ax630c/vocos_full.axmodel
```

当前 AX630C 入口脚本：

```text
scripts/ax630c/01_export_static_onnx.sh
scripts/ax630c/02_infer_onnx.sh
scripts/ax630c/03_generate_calib.sh
scripts/ax630c/04_quant.sh
```

## 2. 工程目录

量化工程：

```bash
cd /data/shared/huyuan/TTS_quant/ZipVoice.AXERA
```

板端 AXCL 模型目录：

```text
/data/shared/huyuan/TTS/ZipVoice.AXERA_AXCL/models/zipvoice_distill_ax630C
```

当前板端目录只保留运行需要的文件：

```text
decoder4_split_manifest.json
decoder_part0.axmodel
decoder_part1.axmodel
decoder_part2.axmodel
decoder_part3_0.axmodel
decoder_part3_1.axmodel
decoder_part3_2.axmodel
decoder_part3_3.axmodel
encoder.axmodel
encoder_core.onnx
fm_decoder_part0.onnx
fm_decoder_part3.onnx
runtime_config.json
```

量化工程内也保留一份同结构最终 package：

```text
model_convert/zipvoice_distill_ax630c/decoder4_split_manifest.json
model_convert/zipvoice_distill_ax630c/encoder.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part0.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part1.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part2.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part3_0.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part3_1.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part3_2.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part3_3.axmodel
```

`decoder4_split_manifest.json` 当前包含 7 个 decoder part：

```text
fm_decoder_part0
fm_decoder_part1
fm_decoder_part2
fm_decoder_part3_0
fm_decoder_part3_1
fm_decoder_part3_2
fm_decoder_part3_3
```

## 3. 环境

导出 ONNX、ONNX Runtime 验证和校准数据生成使用 `moss-quant`：

```bash
source /data/huyuan/miniforge3/etc/profile.d/conda.sh
conda activate moss-quant
cd /data/shared/huyuan/TTS_quant/ZipVoice.AXERA
```

Pulsar2 量化使用 `base`，并激活 NPU 环境：

```bash
source /data/huyuan/miniforge3/etc/profile.d/conda.sh
conda activate base
source /data/huyuan/npu-codebase/script/npu_dev
cd /data/shared/huyuan/TTS_quant/ZipVoice.AXERA
```

建议所有临时文件和缓存放在工程内：

```bash
export WORK_TMP=$PWD/.work_tmp
export TMPDIR=$WORK_TMP/tmp
export MPLCONFIGDIR=$WORK_TMP/matplotlib
export XDG_CACHE_HOME=$WORK_TMP/xdg_cache
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
```

## 4. 当前保留的 part3 四段文件

### ONNX 生成物

以下文件由 `scripts/ax630c/01_export_static_onnx.sh` 和 `scripts/ax630c/03_generate_calib.sh` 生成。为便于上传 GitHub，当前仓库不保留这些 ONNX 本体。

```text
models/decoder4_static_distill_1024_ax630c_part3four/fm_decoder_part3_0.onnx
models/decoder4_static_distill_1024_ax630c_part3four/fm_decoder_part3_1.onnx
models/decoder4_static_distill_1024_ax630c_part3four/fm_decoder_part3_2.onnx
models/decoder4_static_distill_1024_ax630c_part3four/fm_decoder_part3_3.onnx
models/decoder4_static_distill_1024_ax630c_part3four/part3_four_split_manifest.json
```

### 脚本

```text
utils/model_convert/split_fm_decoder_part3_four.py
utils/model_convert/generate_part3four_calib.py
utils/model_convert/make_part3four_quant_configs.py
```

### 校准数据

```text
model_convert/calib_data_decoder4_rich_distill_1024_part3four
```

### Pulsar2 config

```text
model_convert/config_fm_decoder_part3_0_check0-distill-rich-four-u16-powu8-ax630c.json
model_convert/config_fm_decoder_part3_1_check0-distill-rich-four-u16-powu8-ax630c.json
model_convert/config_fm_decoder_part3_2_check0-distill-rich-four-u16-powu8-ax630c.json
model_convert/config_fm_decoder_part3_3_check0-distill-rich-four-u16-powu8-ax630c.json
```

### 构建日志

```text
model_convert/build-fm_decoder_part3_0-check0-distill-rich-four-u16-powu8-ax630c.log
model_convert/build-fm_decoder_part3_1-check0-distill-rich-four-u16-powu8-ax630c.log
model_convert/build-fm_decoder_part3_2-check0-distill-rich-four-u16-powu8-ax630c.log
model_convert/build-fm_decoder_part3_3-check0-distill-rich-four-u16-powu8-ax630c.log
```

说明：`build-fm_decoder_part3_{0..3}-...` 大缓存目录已清理。最终 axmodel 已集中保留在 `model_convert/zipvoice_distill_ax630c`。

### 板端 manifest 备份

量化工程内保留一份四段版 AXCL manifest：

```text
model_convert/decoder4_split_manifest_part3four.json
```

## 5. part3 四段拆分结构

拆分点：

```text
/fm_decoder/4/0/bypass/Add_output_0
/fm_decoder/4/1/bypass/Add_output_0
/fm_decoder/4/2/bypass/Add_output_0
```

四段输入输出：

| 段 | 输入 | 输出 |
|---|---|---|
| part3_0 | `decoder_hidden_p2`, `time_hidden`, `padding_mask` | `decoder_hidden_p30` |
| part3_1 | `decoder_hidden_p30`, `time_hidden`, `padding_mask` | `decoder_hidden_p31` |
| part3_2 | `decoder_hidden_p31`, `time_hidden`, `padding_mask` | `decoder_hidden_p32` |
| part3_3 | `decoder_hidden_p32`, `time_hidden`, `padding_mask` | `v` |

AXCL Python 和 C++ runtime 都按 `decoder4_split_manifest.json` 的 `decoder_parts` 顺序串联执行，因此四段版不需要额外新增 runtime 分支。

## 6. 量化配置要点

part3 四段版使用 check0 构建，目的是快速评估音频质量。当前主观效果优于八段版，所以暂定为板端可用版本。

共同配置：

- `target_hardware: AX620E`
- `npu_mode: NPU2`
- `compiler.check: 0`
- `enable_onnxsim: true`
- 默认 `U16`
- `Pow` 使用 `U8`
- `conv_bias_data_type: FP32`
- `precision_analysis: true`
- `precision_analysis_method: EndToEnd`
- `disable_auto_refine_scale: true`
- `transformer_opt_level: 0`

注意：

- `Pow` 在 AX630C 上按已有经验保留 U8。
- 不再保留大范围 HardSoftplus FP32 试验配置；该方向曾出现后端风险，且没有稳定改善最终音质。
- 不再保留 8 段拆分配置；虽然局部 cosine 更高，但实际音频主观效果不如四段。

## 7. 复现命令

### 7.1 重新拆分 part3 ONNX

```bash
source /data/huyuan/miniforge3/etc/profile.d/conda.sh
conda activate moss-quant
cd /data/shared/huyuan/TTS_quant/ZipVoice.AXERA

python3 utils/model_convert/split_fm_decoder_part3_four.py
```

输出：

```text
models/decoder4_static_distill_1024_ax630c_part3four/
```

### 7.2 生成四段 part3 校准数据

```bash
source /data/huyuan/miniforge3/etc/profile.d/conda.sh
conda activate moss-quant
cd /data/shared/huyuan/TTS_quant/ZipVoice.AXERA

python3 utils/model_convert/generate_part3four_calib.py
```

输出：

```text
model_convert/calib_data_decoder4_rich_distill_1024_part3four/
```

### 7.3 生成 Pulsar2 config

```bash
cd /data/shared/huyuan/TTS_quant/ZipVoice.AXERA
python3 utils/model_convert/make_part3four_quant_configs.py
```

输出：

```text
model_convert/config_fm_decoder_part3_0_check0-distill-rich-four-u16-powu8-ax630c.json
model_convert/config_fm_decoder_part3_1_check0-distill-rich-four-u16-powu8-ax630c.json
model_convert/config_fm_decoder_part3_2_check0-distill-rich-four-u16-powu8-ax630c.json
model_convert/config_fm_decoder_part3_3_check0-distill-rich-four-u16-powu8-ax630c.json
```

### 7.4 量化四段 part3

```bash
source /data/huyuan/miniforge3/etc/profile.d/conda.sh
conda activate base
source /data/huyuan/npu-codebase/script/npu_dev
cd /data/shared/huyuan/TTS_quant/ZipVoice.AXERA

pulsar2 build \
  --config model_convert/config_fm_decoder_part3_0_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p2:1024x1x512;time_hidden:1x192;padding_mask:1x1024"

pulsar2 build \
  --config model_convert/config_fm_decoder_part3_1_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p30:1024x1x512;time_hidden:1x192;padding_mask:1x1024"

pulsar2 build \
  --config model_convert/config_fm_decoder_part3_2_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p31:1024x1x512;time_hidden:1x192;padding_mask:1x1024"

pulsar2 build \
  --config model_convert/config_fm_decoder_part3_3_check0-distill-rich-four-u16-powu8-ax630c.json \
  --input_shapes "decoder_hidden_p32:1024x1x512;time_hidden:1x192;padding_mask:1x1024"
```

生成后需要复制到最终 package：

```text
model_convert/zipvoice_distill_ax630c/decoder_part3_0.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part3_1.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part3_2.axmodel
model_convert/zipvoice_distill_ax630c/decoder_part3_3.axmodel
```

## 8. 部署到 AXCL 板端目录

目标目录：

```text
/data/shared/huyuan/TTS/ZipVoice.AXERA_AXCL/models/zipvoice_distill_ax630C
```

需要复制：

```text
decoder_part3_0.axmodel
decoder_part3_1.axmodel
decoder_part3_2.axmodel
decoder_part3_3.axmodel
decoder4_split_manifest.json
```

当前板端目录已经完成替换，并清理了旧单体、两段和八段 part3 文件。

## 9. 板端测试命令

段落测试：

```bash
cd /mnt/huyuan/workspace/TTS/ZipVoice.AXERA_AXCL

python3 infer_zipvoice_axera.py \
  --model-name zipvoice_distill_ax630C \
  --text-file assets/paragraphs/zh_ginkgo.txt \
  --prompt-text "不管怎么样我和汤姆还是要感谢贝尔卡金的援手" \
  --prompt-wav assets/moss_prompts/zh_1_4p5s.wav \
  --output-wav outputs/zh_ginkgo_distill_ax630C_part3four.wav \
  --seed 42
```

单句测试：

```bash
cd /mnt/huyuan/workspace/TTS/ZipVoice.AXERA_AXCL

python3 infer_zipvoice_axera.py \
  --model-name zipvoice_distill_ax630C \
  --text "今天午后天气很好，我打开窗户，听见远处有人聊天，水杯也轻轻晃了一下。" \
  --prompt-text "不管怎么样我和汤姆还是要感谢贝尔卡金的援手" \
  --prompt-wav assets/moss_prompts/zh_1_4p5s.wav \
  --output-wav outputs/zh_sentence_distill_ax630C_part3four.wav \
  --seed 42
```

## 10. 后续如果继续优化

如果四段版仍有轻微噪声，建议按下面顺序处理：

1. 不优先回到八段。八段 cosine 高但主观音质不占优。
2. 先固定四段结构，逐段对比 ONNX 与 axmodel 输出，定位哪一段导致音质波动。
3. 对低精度段只做窄范围 FP32：
   - 优先看 `conv_module2/depthwise_conv`。
   - 再看 HardSoftplus 内部的局部算子。
   - 避免把整段 HardSoftplus 或大子图直接 FP32，之前这类方案有后端失败风险。
4. 如果要跑最终严格验收，再从当前四段 check0 方案迁移到 check2；check2 失败时按失败 tensor 做定位，不要盲目重复构建。

check2 失败时的处理原则：

- 先看失败类型：前端 quant 精度低、IR transformation 失败、NPU graph check 失败、输入 dtype/shape 不匹配，还是后端 compile crash。
- 如果是 `check npu graph [subgraph_npu_*] failed`，但 quant ONNX EndToEnd 精度正常，优先怀疑后端 lowering/NPU 实现问题。
- 对 `GatherElements`、动态 reshape/slice、Where+Softmax、大范围激活 outlier，优先考虑 ONNX 图替换或更合理拆分，而不是持续扩大 FP32。
- 子图 FP32 规则如果没有覆盖 producer，需要单独指定 producer layer，或者把 start tensor 前移。
