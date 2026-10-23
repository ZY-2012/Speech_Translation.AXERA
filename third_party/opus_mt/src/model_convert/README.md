# 模型转换：OPUS-MT → ONNX → Pulsar2 U16

## 0. 目录约定

```bash
export OPUS_MT_ROOT=/path/to/Opus-MT.AXERA
cd $OPUS_MT_ROOT
```

脚本把工作文件放在 `$OPUS_MT_ROOT/<direction>/`：

```
origin/zh-en/        # HuggingFace 浮点权重（需自行下载，不入库）
origin/en-zh/
zh-en/export/        # encoder.onnx / decoder.onnx / calib_data
zh-en/compile/       # encoder.axmodel / decoder.axmodel
en-zh/export/ ...
en-zh/compile/ ...
```

## 1. 下载浮点模型

```bash
hf download Helsinki-NLP/opus-mt-zh-en --local-dir origin/zh-en
hf download Helsinki-NLP/opus-mt-en-zh --local-dir origin/en-zh
```

需要 `config.json`、`pytorch_model.bin`、`source.spm`、`target.spm`、`vocab.json`。
国内网络可用 `HF_ENDPOINT=https://hf-mirror.com`。

## 2. 导出静态 ONNX + 对分

```bash
python model_convert/export_models.py --direction zh-en
python model_convert/export_models.py --direction en-zh
```

- encoder：`input_ids[1,128] int32` + `attention_mask[1,128] fp32`
- decoder：左对齐前缀，`decoder_input_ids[1,64]` + `decoder_attention_mask[1,64]` +
  `position_ids[1,64]` + `encoder_hidden_states[1,128,512]` + `encoder_attention_mask[1,128]`
- 正式产物先过 `onnxsim` 再 `onnxslim`
- 自动做 ONNX vs PyTorch 对分和 4 句端到端贪心对分；通过标准：
  cosine 1.000000、贪心输出完全一致

> 关键坑：HF `MarianMTModel.forward` 在 `lm_head` 之后还要加 checkpoint 里的
> `final_logits_bias`。漏掉这一步，单步 cosine 看起来正常，但自回归几 token 后就会发散。

## 3. 生成校准集

```bash
python model_convert/prepare_calib.py --direction zh-en
python model_convert/prepare_calib.py --direction en-zh
```

- 数据：`model_convert/data/<direction>.test.txt`（Tatoeba test 的 200 对样本，
  每 4 行一组：source、ref1、ref2、空行）
- encoder：64 句，padding 到 128
- decoder：每句取 4 个真实目标前缀（首步 / 1/2 / 3/4 / 末步），共 256 个样本
- decoder 的 `encoder_hidden_states` 来自 FP32 encoder 真实输出
- 每个输入一张 tar.gz：`input_ids`、`attention_mask`、`encoder_hidden_states`、
  `decoder_input_ids`、`decoder_attention_mask`、`position_ids`、
  `decoder_encoder_attention_mask`

## 4. Pulsar2 U16 量化

先准备 Pulsar2 Docker 镜像（Pulsar2 ≥ 6.0），通过 `PULSAR2_IMAGE` 指定：

```bash
PULSAR2_IMAGE=your-pulsar2-image \
python model_convert/compile_models.py --direction zh-en --part both --u16
PULSAR2_IMAGE=your-pulsar2-image \
python model_convert/compile_models.py --direction en-zh --part both --u16
```

U16 配置等价于：

```json
"quant": {
  "layer_configs": [
    {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"],
     "data_type": "U16"}
  ],
  "highest_mix_precision": false
}
```

权重保持 S8。INT8 全量量化时 decoder logits cosine 只有 0.971；U16 后 zh-en 0.9996、
en-zh 0.9972，200 句端到端 BLEU/chrF 与 FP32 基本持平。

产物：

```
zh-en/compile/encoder.axmodel   83 MiB
zh-en/compile/decoder.axmodel  124 MiB
```

## 5. 精度验收（板端）

量化产物直接上板，用 axengine 跑 20 组真实输入：

| 方向 | encoder mean/min | decoder logits mean/min | next-token argmax |
|------|------------------|--------------------------|-------------------|
| zh-en | 0.999841 / 0.999808 | 0.999580 / 0.998458 | 20/20 |
| en-zh | 0.999548 / 0.998712 | 0.997155 / 0.979991 | 20/20 |

200 句 Tatoeba：

| 方向 | BLEU (板端 / FP32) | chrF (板端 / FP32) | 完全一致 |
|------|--------------------|--------------------|----------|
| zh-en | 32.78 / 33.48 | 53.91 / 54.73 | 179/200 |
| en-zh | 28.72 / 28.63 | 24.65 / 24.49 | 168/200 |

## 依赖

```bash
pip install -r model_convert/requirements.txt
```

Pulsar2 需要 Docker 和有效 license（按 AXERA Pulsar2 发布环境配置）。
