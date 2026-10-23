# ZipVoice.AXERA

ZipVoice / ZipVoice distill 的静态 ONNX 导出、ONNX 推理验证、校准数据生成和 Pulsar2 量化工程，以及板端 C++ 推理源码。

量化后的 axmodel 与推理可执行文件发布在
[HuggingFace: AXERA-TECH/ZipVoice.AXERA](https://huggingface.co/AXERA-TECH/ZipVoice.AXERA)。

当前入口脚本按用途拆分：

```text
scripts/
├── ax650/
│   ├── 01_export_static_onnx.sh
│   ├── 02_infer_onnx.sh
│   ├── 03_generate_calib.sh
│   └── 04_quant.sh
├── ax630c/
│   ├── 01_export_static_onnx.sh
│   ├── 02_infer_onnx.sh
│   ├── 03_generate_calib.sh
│   └── 04_quant.sh
└── vocoder/
    ├── 01_export_onnx.sh
    ├── 02_infer_onnx.sh
    ├── 03_generate_calib.sh
    ├── 04_quant_ax650_check2.sh
    └── 05_quant_ax630c_full.sh
```

## 原始模型

导出前请先下载原始 PyTorch checkpoint，并放到当前工程的 `models/` 目录。

模型下载地址：

```text
https://huggingface.co/k2-fsa/ZipVoice
```

建议目录：

```text
models/zipvoice/
models/zipvoice_distill/
models/vocos-mel-24khz/
```

## 环境

ONNX 导出、ONNX Runtime 推理和校准数据生成使用 `zipvoice-quant`：

```bash
conda create -n zipvoice-quant python==3.12
conda activate zipvoice-quant
pip install -r requirements.txt
```


## AX650 ZipVoice


```bash
# 1. 导出标准 ZipVoice + distill 静态 ONNX
bash scripts/ax650/01_export_static_onnx.sh

# 2. ONNX 推理验证
bash scripts/ax650/02_infer_onnx.sh

# 3. 生成 rich 校准数据
bash scripts/ax650/03_generate_calib.sh

激活 Pulsar2 量化环境，后续模型量化也是一样步骤
# 4. Pulsar2 量化
bash scripts/ax650/04_quant.sh
```

输出：

```text
model_convert/zipvoice_ax650/
model_convert/zipvoice_distill_ax650/
```

## AX630C ZipVoice distill 模型


```bash
# 1. 导出 AX630C-friendly distill 静态 ONNX
bash scripts/ax630c/01_export_static_onnx.sh

# 2. ONNX 推理验证
bash scripts/ax630c/02_infer_onnx.sh

# 3. 生成 rich 校准数据，并生成 part3 四段校准数据
bash scripts/ax630c/03_generate_calib.sh

# 4. 量化 part3 四段并更新最终 package
bash scripts/ax630c/04_quant.sh
```

当前最终 package：

```text
model_convert/zipvoice_distill_ax630c/
├── decoder4_split_manifest.json
├── encoder.axmodel
├── decoder_part0.axmodel
├── decoder_part1.axmodel
├── decoder_part2.axmodel
├── decoder_part3_0.axmodel
├── decoder_part3_1.axmodel
├── decoder_part3_2.axmodel
└── decoder_part3_3.axmodel
```


## Vocoder

vocoder 独立量化，mel → waveform，不依赖 ZipVoice encoder/decoder。

公共 ONNX/校准流程：

```bash
# 1. 导出 Vocos ONNX
bash scripts/vocoder/01_export_onnx.sh

# 2. ONNX 推理验证
bash scripts/vocoder/02_infer_onnx.sh

# 3. 生成 vocoder 校准数据
bash scripts/vocoder/03_generate_calib.sh
```

AX650 vocoder：

```bash
bash scripts/vocoder/04_quant_ax650_check2.sh
```

输出：

```text
model_convert/vocoder_ax650/vocos_full.axmodel
```

AX630C vocoder：

```bash
bash scripts/vocoder/05_quant_ax630c_full.sh
```

输出：

```text
model_convert/vocoder_ax630c/vocos_full.axmodel
```

## 板端 C++ 推理

源码在 `cpp/`，编译与运行说明见 `cpp/README.md`：

```bash
bash cpp/build_ax650.sh    # → cpp/install/ax650/zipvoice
bash cpp/build_ax630c.sh   # → cpp/install/ax630c/zipvoice
bash cpp/build_axcl.sh     # → cpp/install/axcl/zipvoice
```

工具链（gcc 交叉编译器与 BSP SDK）下载方式见 `cpp/download_bsp.sh` 与 `cpp/README.md`。

## 板端 Python 推理

Python 推理脚本（axengine）见 HuggingFace 工程 `infer_zipvoice_axera.py` 与 `run_ax*.sh`。


参考：

[ZipVoice](https://huggingface.co/k2-fsa/ZipVoice)
