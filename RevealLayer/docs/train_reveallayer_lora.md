# RevealLayer LoRA 训练说明

这套训练脚本按 RevealLayer 论文和官方推理权重结构组织：FLUX transformer 使用 LoRA 微调，`layer_pe` 与 `refiner` 作为额外完整参数训练并单独保存，`L_alpha` 与 `L_orth` 通过 transparent decoder 接到单步 Flow Matching 训练中。

## 参考来源

- RevealLayer 推理代码：复用 `[full image, background, foreground layers...]`、`validation_box`、`latent_image_ids`、Region-Aware Attention、OGA/refiner 输入，以及 transparent decoder 的 latent 尺度。
- DiffSynth FLUX 训练代码：对齐 Flow Matching 约定，即 `x_t = (1 - sigma) * x_0 + sigma * eps`，target velocity 为 `eps - x_0`。
- RevealLayer 论文：总目标为 `L = L_FM + lambda_alpha L_alpha + lambda_o L_orth`，并使用 hard alpha loss 与 soft orthogonality loss 改善边界和图层残留。
- Qwen-Image-Layered：只借鉴 RGBA 图层目标和可变图层数量的数据组织思路，不搬用 Qwen 的 processor、tokenizer 或 pipeline。

## 权重结构

训练保存格式保持和官方推理代码一致：

- `pytorch_lora_weights.safetensors`：FLUX transformer LoRA。
- `layer_pe.pt`：`transformer.layer_pe` 的完整 tensor。
- `Refiner.pt`：`transformer.refiner` 的完整 state dict，不带 `refiner.` 前缀。
- `transparent_decoder_ckpt.pth`：仅在 `transparent_decoder.train: true` 时保存。

默认配置会训练 LoRA、`layer_pe` 和 `refiner`。transparent decoder 默认冻结，只作为可微 RGBA/alpha 监督头。

## 输出目录

默认所有训练输出都写到：

```text
outputs/reveallayer_lora/
```

目录结构如下：

```text
outputs/reveallayer_lora/
  resolved_config.json
  logs/
  checkpoint-1000/
    pytorch_lora_weights.safetensors
    layer_pe.pt
    Refiner.pt
  previews/
    step-001000/
      source_full_image.png
      pred_layer_00_full_rgba.png
      pred_layer_01_background_rgba.png
      pred_layer_02_foreground_rgba.png
      target_layer_01_background_rgba.png
      target_layer_02_foreground_rgba.png
      pred_merged_rgba.png
```

`checkpointing_steps` 控制权重保存频率，`preview_steps` 控制可视化保存频率。默认两者都是 `1000`，因此每次保存 checkpoint 时也会保存一组预览图。

## 数据 Manifest

支持 JSON 或 JSONL。JSON 读取使用 `utf-8-sig`，可以兼容 PowerShell 导出的 BOM 文件。

```json
{
  "imgid": "sample_0001",
  "full_image": "sample_0001/full_image.png",
  "background": "sample_0001/background.png",
  "LayerInfoRaw": [
    "sample_0001/layer_0.png",
    "sample_0001/layer_1.png"
  ],
  "prompt": "a layered decomposition of the scene",
  "detections": [
    {"bbox": [120, 80, 520, 760]},
    {"bbox": [540, 180, 880, 720]}
  ]
}
```

图层顺序固定如下：

1. `layer 0`：完整图像条件层，复制到 noisy latent 中，并从 FM loss 中排除。
2. `layer 1`：背景目标层。
3. `layer 2+`：前景 RGBA 目标层，按对应 bbox 裁剪参与训练。

如果样本缺少 `background` 字段，数据集会用完整图像作为背景目标的工程 fallback。这只是为了让数据管线可运行，不是论文设定。

`LayerInfoRaw` 与 `detections[*].bbox` 必须一一对应。脚本也兼容 `layers`、`target_layers`、`rgba_layers`、`foreground_layers` 和 `layout`、`bboxes`、`boxes` 等别名。

如果数据集中 `layer_0` 的上下堆叠语义不统一，可以通过 `data.layer_order` 控制读取顺序：

```yaml
data:
  layer_order: as_is   # 按 manifest 中的 LayerInfoRaw/detections 原始顺序
  # layer_order: reverse  # 同时反转 layer 路径和对应 bbox
```

如果某批数据中 `LayerInfoRaw[0]` 表示最上层，通常保持 `as_is`；如果它表示最下层，则用 `reverse`。

## Loss 链路

训练仍然是单步 diffusion/FM 训练，不需要跑完整 denoising 采样过程。

```text
x_t -> transformer -> v_pred
x_0_pred = x_t - sigma * v_pred
x_0_pred -> transparent decoder -> pred RGB / pred alpha
```

`L_FM`：

```text
target = eps - x_0
L_FM = MSE(v_pred, target)
```

`L_alpha`：

```text
delta = tau * |alpha_pred - alpha_gt|
L_alpha = - mean(delta^gamma * log(1 - delta + eps))
```

默认 `tau=0.95`、`gamma=1.5`，只在前景层 `layer 2+` 上计算。

`L_orth`：

使用像素空间 cosine-style 约束，比较预测背景/前景在前景 bbox 区域内的相关性，并以 GT 图层相关性作为 soft target，用于抑制背景中的前景残留和图层串扰。

## 启动命令

单卡训练：

```powershell
accelerate launch train_reveallayer_lora.py --config configs/train_reveallayer_lora.yaml
```

从最新 checkpoint 恢复：

```powershell
accelerate launch train_reveallayer_lora.py --config configs/train_reveallayer_lora.yaml --resume_from_checkpoint latest
```

最小 smoke test：

```powershell
accelerate launch train_reveallayer_lora.py --config configs/train_reveallayer_lora.yaml --max_train_steps 1 --checkpointing_steps 1
```

## 关键配置

```yaml
lora:
  train_layer_pe: true
  train_refiner: true

transparent_decoder:
  checkpoint: ckpts/xvae/transparent_decoder_ckpt.pth
  train: false

loss:
  enable_alpha_loss: true
  enable_orth_loss: true
```

如果要一起微调 transparent decoder，将 `transparent_decoder.train` 改为 `true`。这样会增加显存消耗，并且 checkpoint 中会额外保存 `transparent_decoder_ckpt.pth`。
