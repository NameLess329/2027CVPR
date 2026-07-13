# 阶段 1 RCM 实现说明

本次实现对应 `token_contribution_方案整理与逐步实现计划.md` 中的阶段 1：只做 Reference Contribution Map, RCM 的监督、预测、训练和可视化，不改变 RevealLayer 的生成结果。

## 实现边界

阶段 1 做：

- 从 RGBA 前景层和全局图层顺序构造 `alpha_visible`。
- 用 `alpha_visible` 生成 token 级 `rcm_target`。
- 用 patch 内 alpha 方差生成 `rcm_conf`。
- 在 refiner 前旁路预测 RCM logits。
- RCM loss 只训练 `RCMHead`。
- 导出 target、confidence、prediction、error 和 overlay 可视化。

阶段 1 不做：

- 不修改 MM-DiT attention mask。
- 不用 RCM 加权 `clean_tokens`。
- 不用 gamma 调制 refiner delta。
- 不训练 FLUX 主干、LoRA、`layer_pe`、`refiner` 或 transparent decoder。
- 不预测 GCM。

## 新增和修改文件

新增文件：

- `models/rcm.py`
  - 定义 `RCMHead` 和 `RCMHeadOutput`。
  - 输入为当前层 hidden tokens、bbox 内 clean condition tokens、bbox、timestep embedding 和 token 位置。
  - 输出每个有效层的 RCM logits。

- `training/rcm_targets.py`
  - `visible_alpha_from_order`：用 alpha-over 规则构造可见 alpha。
  - `area_target_and_conf`：area downsample 到 16 像素 token grid，并用 patch 方差生成置信度。
  - `build_rcm_targets_from_alpha`：从现有 LoRA 训练 batch 的 `target_alpha` 和 `validation_boxes` 生成 RCM target。
  - `weighted_bce_with_logits`、`rcm_metrics`：阶段 1 loss 和指标。

- `tools/build_rcm_stage1_supervision.py`
  - 从 RGBA 图层 manifest 生成离线监督数据。
  - 输出 `manifest_rcm_stage1.jsonl`、`npz/` 和 `vis/`。

- `tools/visualize_rcm_stage1.py`
  - 对监督或训练预测 `.npz` 重新导出 heatmap。

- `train_rcm_stage1.py`
  - 复用现有 `docs/train_reveallayer_lora.md` 描述的 pipeline、dataset 和单步 flow-matching 前向。
  - 主模型全部冻结，optimizer 只包含 `rcm_head.parameters()`。
  - 每隔 `preview_steps` 导出 RCM 可视化。

- `configs/build_rcm_stage1_supervision.yaml`
- `configs/train_rcm_stage1.yaml`

修改文件：

- `models/custom_model_mmdit.py`
  - 增加 `set_rcm_probe(...)` 可选挂钩。
  - 默认关闭，不改变原有 forward 输出。
  - 开启后在 `clean_tokens = crop_each_layer(adapter_data, list_layer_box[1:], 16)` 之后、`refiner.forward_feature_attn(...)` 之前计算 RCM side output。
  - RCM 输入全部 `detach()`，阶段 1 loss 不反传主生成路径。

## 与现有训练脚本的对齐

现有 LoRA 训练文档里约定：

- `validation_boxes = [full image, background, foreground layers...]`
- `target_alpha = [background, foreground layers...]`
- `LayerInfoRaw` 和 `detections[*].bbox` 一一对应。
- `layer_order: as_is/reverse` 决定前景层读取顺序。
- bbox 在数据集里已经按 16 像素对齐。

本实现直接复用这些约定。RCM 训练时默认只监督前景层，即 `validation_boxes[2:]` 和 `target_alpha[1:]`。如果需要背景 RCM，可在 `configs/train_rcm_stage1.yaml` 中设置：

```yaml
rcm:
  include_background: true
```

## 运行命令

生成离线 RCM 监督：

```powershell
python tools/build_rcm_stage1_supervision.py --config configs/build_rcm_stage1_supervision.yaml
```

训练在线 RCM probe：

```powershell
accelerate launch train_rcm_stage1.py --config configs/train_rcm_stage1.yaml
```

最小 smoke test：

```powershell
accelerate launch train_rcm_stage1.py --config configs/train_rcm_stage1.yaml --max_train_steps 1 --preview_steps 1 --checkpointing_steps 1
```

重新可视化某个 `.npz`：

```powershell
python tools/visualize_rcm_stage1.py --npz outputs/rcm_stage1/previews/step-000001/rcm_predictions.npz --output_dir outputs/rcm_stage1/previews/step-000001/revisualized
```

## 输出

离线监督默认输出：

```text
supervision/rcm_stage1/
  manifest_rcm_stage1.jsonl
  npz/
  vis/
```

在线训练默认输出：

```text
outputs/rcm_stage1/
  resolved_config.json
  logs/
  checkpoint-000001/
    rcm_head.pt
  previews/
    step-000001/
      input.png
      layer_02_target.png
      layer_02_conf.png
      layer_02_pred.png
      layer_02_error.png
      layer_02_pred_overlay.png
      rcm_predictions.npz
```

## 验收标准

阶段 1 通过时应看到：

- RCM target 在当前图层可见区域高。
- 后层被前层遮挡的位置 target 低。
- 边界 token 的 `rcm_conf` 低于纯区域。
- 均匀半透明区域允许软 target，且 confidence 不应被错误压低。
- `train_rcm_stage1.py` 的 checkpoint 只保存 `rcm_head.pt`。
- 开启 RCM probe 后，生成路径没有使用 RCM logits，因此不会改变 baseline 生成结果。

## 后续阶段入口

阶段 2 如果要让 RCM 影响生成，建议只在 `models/custom_model_mmdit.py` 的 refiner 前新增可选分支：

```python
clean_tokens = clean_tokens * rcm_prob
```

但当前阶段没有做这一步。现在的 RCM 是可解释 probe，用来确认 token-level reliable visible reference 能被单独学出来。
