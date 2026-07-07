# Qwen Image Layered

这个仓库从 `DiffSynth-Studio` 中整理出 `Qwen-Image-Layered` 训练与推理所需的核心代码，只保留这一条方法链路相关的脚本、模型管线和基础训练组件。

## 项目结构

```text
Qwen_Image_Layered/
├── README.md
├── scripts/
│   ├── Qwen-Image-Layered.py
│   ├── Qwen-Image-Layered.sh
│   └── train.py
└── src/
    └── diffsynth/
        ├── configs/
        ├── core/
        ├── diffusion/
        ├── models/
        ├── pipelines/
        └── utils/
```

## 包含内容

- `scripts/Qwen-Image-Layered.py`：Qwen-Image-Layered 推理示例。
- `scripts/Qwen-Image-Layered.sh`：LoRA 训练启动脚本。
- `scripts/train.py`：训练入口。
- `src/diffsynth/`：训练和推理依赖到的 `DiffSynth` 子集，只保留 `Qwen-Image-Layered` 相关管线与通用底座。

## 运行前准备

建议在项目根目录执行命令，并安装训练/推理需要的依赖，例如：

```bash
pip install torch torchvision accelerate transformers peft modelscope pillow einops tqdm safetensors packaging
```

如果你的环境不会自动把 `src` 加入 Python 路径，也没关系，`scripts/train.py` 和 `scripts/Qwen-Image-Layered.py` 已经在脚本内部处理了本地导入路径。

## 推理

在项目根目录运行：

```bash
python scripts/Qwen-Image-Layered.py
```

脚本会：

- 下载示例输入图到 `data/example_image_dataset/`
- 加载 `Qwen/Qwen-Image-Layered`、`Qwen/Qwen-Image` 和 `Qwen/Qwen-Image-Edit`
- 将生成结果保存到 `outputs/inference/`

## 训练

在 Linux / WSL / Git Bash 环境下运行：

```bash
bash scripts/Qwen-Image-Layered.sh
```

这个脚本会：

- 下载 `DiffSynth-Studio/diffsynth_example_dataset` 中 `qwen_image/Qwen-Image-Layered` 数据
- 调用 `accelerate launch scripts/train.py`
- 将 LoRA 输出保存到 `models/train/Qwen-Image-Layered_lora`

如果你想手动运行训练命令，可以直接执行：

```bash
accelerate launch scripts/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Layered \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Layered/metadata.json \
  --data_file_keys "image,layer_input_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Layered:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Layered_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --extra_inputs "layer_num,layer_input_image" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
```

## 说明

- 这里保留了 `Qwen-Image-Layered` 所需的通用训练基础设施，比如 `core/` 和 `diffusion/`，因为训练入口对这些模块有直接依赖。
- `configs/model_configs.py` 和 `configs/vram_management_module_maps.py` 已裁剪为 `Qwen-Image-Layered` 相关内容，避免混入其他模型注册信息。
