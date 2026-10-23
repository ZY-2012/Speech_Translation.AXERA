# Python 板端推理

纯 Python 运行时，依赖仅 `numpy + sentencepiece + pyaxengine`，不需要
transformers / torch / onnxruntime。

## 安装

```bash
pip install -r requirements.txt
```

## 用法

```bash
# 单句
python python/translate.py --direction zh-en --model-root models \\
    --text "这是一个测试。" --stats

# 批量（每行一句）
python python/translate.py --direction en-zh --model-root models \\
    --input input.txt --output output.txt --stats
```

`en-zh` 会自动补 `>>cmn_Hans<<`；其他中文变体可自己传前缀。

## 作为库

```python
import sys
sys.path.insert(0, "python")
from opus_mt_ax import Translator

tr = Translator("zh-en", model_root="models")       # 或 "en-zh"
print(tr.translate("这是一个测试。"))
out, stats = tr.translate("这是一个测试。", return_stats=True)
print(stats)
```

`Translator` 初始化时会预热 2 次；`stats` 字段：

```
src_tokens, out_tokens, enc_ms, dec_ms, dec_step_ms, truncated
```

## 模型目录

```
models/zh-en/
  encoder.axmodel  decoder.axmodel
  vocab.json  source.spm  target.spm  model_meta.json
models/en-zh/ ...
```

从 [HY-2012/Opus-MT.AXERA](https://huggingface.co/HY-2012/Opus-MT.AXERA) 下载。
