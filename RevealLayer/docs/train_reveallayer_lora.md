# RevealLayer LoRA Training

This repo originally exposes an inference path. The training entry added here follows the same layer contract used by `infer.py`:

1. layer 0: full input image
2. layer 1: background
3. layer 2 and later: foreground RGBA layers from the bbox list

The implementation is intentionally conservative. It reproduces the paper's trainable path around FLUX.1-dev, LoRA rank 64, Region-Aware Attention, Occlusion-Guided Adapter data construction, and the flow-matching objective. The paper also defines hard alpha and soft orthogonality losses; helper functions are provided in `training/losses.py`, but the default training loop keeps them disabled until the transparent decoder supervision path is verified with the exact training data.

## Files

- `train_reveallayer_lora.py`: Accelerate training entry.
- `training/dataset.py`: JSON/JSONL manifest loader aligned with `infer.py` preprocessing.
- `training/train_step.py`: differentiable flow-matching forward path derived from `CustomFluxPipelineCfg.__call__`.
- `training/lora.py`: LoRA injection, trainable module selection, and safetensors export.
- `training/losses.py`: flow matching, hard alpha, and soft orthogonality loss helpers.
- `configs/train_reveallayer_lora.yaml`: default training configuration.

## Data Format

The loader accepts the benchmark-style JSON used by inference:

```json
[
  {
    "imgid": "00027",
    "full_image": "./RevealLayerBench/00027/full_image.png",
    "background": "./RevealLayerBench/00027/background.png",
    "LayerInfoRaw": ["./RevealLayerBench/00027/layer_0.png"],
    "detections": [{"bbox": [315, 244, 715, 767]}]
  }
]
```

Supported details:

- `full_image` is required.
- `background` is optional but recommended.
- `LayerInfoRaw`, `layers`, or `foreground_layers` can provide foreground RGBA paths.
- `detections[*].bbox` must be `[x1, y1, x2, y2]`.
- Relative paths are resolved against `--data_root`; if omitted, the manifest directory is used.
- JSON files are read with `utf-8-sig` to tolerate PowerShell UTF-8 BOM exports.

## Paper Settings Reflected Here

From `RevealLayer.pdf`:

- Base model: FLUX.1-dev.
- Fine-tuning: LoRA.
- LoRA rank: 64.
- Optimizer: Prodigy.
- Learning rate: 1.0.
- Training iterations: 50,000.
- Global batch size: 8.
- Image resize: long side 1024.
- Total objective: `L = L_FM + lambda_alpha L_alpha + lambda_o L_orth`, with both lambdas set to 1.0.

The current transformer code accepts a single `list_layer_box` rather than a batched box structure. For that reason, use `--train_batch_size 1` and set `--gradient_accumulation_steps 8` to match the paper's global batch size of 8.

## Single-GPU Command

Edit `configs/train_reveallayer_lora.yaml` first, especially:

- `pretrained_model_name_or_path`
- `ckpt_dir`
- `train_manifest`
- `data_root`
- `output_dir`

Then run:

```bash
accelerate launch train_reveallayer_lora.py \
  --config configs/train_reveallayer_lora.yaml
```

## Multi-GPU Command

Configure Accelerate once:

```bash
accelerate config
```

Then launch:

```bash
accelerate launch --num_processes 8 train_reveallayer_lora.py \
  --config configs/train_reveallayer_lora.yaml
```

The effective global batch is:

```text
num_processes * train_batch_size * gradient_accumulation_steps
```

## Resume Training

Use the default YAML value:

```yaml
resume_from_checkpoint: latest
```

or pass an explicit checkpoint:

```bash
accelerate launch train_reveallayer_lora.py \
  --config configs/train_reveallayer_lora.yaml \
  --resume_from_checkpoint output/reveallayer_lora/checkpoint-2000
```

Each checkpoint contains the Accelerate state and `pytorch_lora_weights.safetensors`. If `layer_pe` or `refiner` are trainable, their weights are saved separately as `reveallayer_extra_trainables.safetensors`. The final output directory receives the same files.

## Logs

TensorBoard logs are written under:

```text
<output_dir>/<logging_dir>
```

Launch TensorBoard with:

```bash
tensorboard --logdir output/reveallayer_lora/logs
```

## Smoke Checks

These checks do not start real training:

```bash
python -m py_compile train_reveallayer_lora.py training/dataset.py training/lora.py training/losses.py training/train_step.py
python train_reveallayer_lora.py --help
```

A one-step smoke run can be done only after local model weights and data paths are valid:

```bash
accelerate launch train_reveallayer_lora.py \
  --config configs/train_reveallayer_lora.yaml \
  --max_train_steps 1 \
  --checkpointing_steps 1
```

## Known Gaps

- The default loop optimizes the flow-matching loss. `L_alpha` and `L_orth` helpers are implemented, but disabled in the YAML because they require confirming the exact transparent-decoder training target and memory cost.
- The custom transformer is not batch-box aware. True per-device batch sizes above 1 require changing `CustomFluxTransformer2DModel.forward` and the ROI mask construction to accept per-sample boxes.
- `pytorch_lora_weights.safetensors` contains only LoRA keys with the `transformer.` prefix expected by `pipeline.load_lora_weights()`. Optional `layer_pe` and `refiner` weights are saved separately and require a small inference-side loader if you choose to train them.
