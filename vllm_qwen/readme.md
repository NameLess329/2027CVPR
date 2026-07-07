# RevealLayer 数据打标脚本

本目录新增 `annotate_reveallayer.py`，用于读取 RevealLayer 多层图像数据集的 `metaData.json`，调用本地 Qwen 多模态模型为每个前景层、背景图和完整合成图生成 prompt，并输出重新设计后的扁平 JSON。

## 输入数据

默认数据集路径：

```text
E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200
```

默认读取：

```text
<dataset-root>\metaData.json
```

原始 `metaData.json` 中每个样本包含：

```json
{
  "imgid": "00027",
  "full_image": "00027/full_image.png",
  "background": "00027/background.png",
  "LayerInfoRaw": ["00027/layer_0.png"],
  "detections": [
    {"bbox": [315, 244, 715, 767]}
  ]
}
```

## 输出格式

脚本默认输出：

```text
<dataset-root>\prompt_annotations.json
```

一个原始样本会被拆成若干条记录，顺序为：

```text
layer_0, layer_1, ..., background, global
```

示例：

```json
[
  {
    "imgid": "00027",
    "item_type": "layer",
    "layer_index": 0,
    "image": "00027/layer_0.png",
    "prompt": "a concise prompt for the foreground object",
    "bbox": [315, 244, 715, 767]
  },
  {
    "imgid": "00027",
    "item_type": "background",
    "layer_index": null,
    "image": "00027/background.png",
    "prompt": "a concise prompt for the clean background scene",
    "bbox": null
  },
  {
    "imgid": "00027",
    "item_type": "global",
    "layer_index": null,
    "image": "00027/full_image.png",
    "prompt": "a concise prompt describing the complete composited image",
    "bbox": [[315, 244, 715, 767]]
  }
]
```

字段说明：

- `imgid`: 原始样本 id。
- `item_type`: `layer`、`background` 或 `global`。
- `layer_index`: 前景层编号；背景和全局图为 `null`。
- `image`: 图像相对 `dataset-root` 的路径。
- `prompt`: Qwen 生成的英文图像 prompt。
- `bbox`: 前景层为对应 bbox；背景为 `null`；全局图为所有前景 bbox 按 layer 顺序组成的列表。

脚本还会保存原始模型回复到：

```text
<output-json>.raw.jsonl
```

如果模型返回的 JSON 无法解析，输出记录会保留空 `prompt`，并附带 `error` 和 `raw_response`，方便后续排查或重跑。

## 推理逻辑

每次推理处理一个样本，将同一个样本的全部图像按如下顺序输入模型：

```text
layer_0, layer_1, ..., background, full_image
```

内置全局 prompt 会要求模型只输出合法 JSON：

```json
{
  "layers": [
    {"prompt": "prompt for layer_0"},
    {"prompt": "prompt for layer_1"}
  ],
  "background": {"prompt": "prompt for the clean background"},
  "global": {"prompt": "prompt for the full composited image"}
}
```

脚本解析模型输出后，再转换为最终的扁平数据格式。

## 运行方法

最小运行命令：

```powershell
python vllm_qwen\annotate_reveallayer.py `
  --model-path E:\path\to\Qwen-model
```

指定输入输出：

```powershell
python vllm_qwen\annotate_reveallayer.py `
  --dataset-root E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200 `
  --metadata E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200\metaData.json `
  --output-json E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200\prompt_annotations.json `
  --model-path E:\path\to\Qwen-model `
  --batch-size 1 `
  --tensor-parallel-size 1
```

先跑少量样本测试：

```powershell
python vllm_qwen\annotate_reveallayer.py `
  --model-path E:\path\to\Qwen-model `
  --limit 5
```

断点续跑：

```powershell
python vllm_qwen\annotate_reveallayer.py `
  --model-path E:\path\to\Qwen-model `
  --resume
```

多卡或多进程分片示例：

```powershell
python vllm_qwen\annotate_reveallayer.py `
  --model-path E:\path\to\Qwen-model `
  --num-shards 2 `
  --shard-index 0 `
  --output-json E:\output\prompt_annotations_shard0.json
```

另一个进程使用 `--shard-index 1` 和不同的 `--output-json`。全部完成后可以再合并 JSON 列表。

## 常用参数

- `--model-path`: 本地 Qwen 多模态模型路径，必填。
- `--batch-size`: vLLM 批大小。多图输入显存占用较高，建议先从 `1` 开始。
- `--max-image-size`: 输入模型前将长边缩放到该尺寸，默认 `1280`；设为 `0` 表示不缩放。
- `--max-new-tokens`: 每个样本最大生成 token 数，默认 `768`。
- `--tensor-parallel-size`: vLLM tensor parallel 大小。
- `--gpu-memory-utilization`: vLLM 显存利用率。
- `--limit`: 只处理前 N 个样本，适合快速测试。
- `--resume`: 跳过输出 JSON 中已经包含 `global` 记录的 `imgid`。
- `--save-every`: 每处理 N 个样本原子写入一次输出 JSON，默认每个样本都保存。

## 环境依赖

需要在可用的 Python/GPU 环境中安装：

```text
Pillow
tqdm
transformers
vllm
```

当前脚本复用了原有 `infer.py` 的 Qwen/vLLM 加载方式：

```python
AutoProcessor.from_pretrained(..., trust_remote_code=True)
LLM(..., trust_remote_code=True)
```
