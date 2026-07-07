# RevealLayer 项目运行说明

本文根据当前仓库文件整理，重点说明：如何配置环境、权重放在哪里、数据放在哪里，以及如何启动推理。

## 1. 项目入口与目录概览

当前项目主要文件：

```text
RevealLayer/
├── README.md
├── requirements.txt
├── infer.py                  # 推理主入口
├── infer.sh                  # 多 GPU 推理脚本，但里面有作者机器的硬编码 Python 路径
├── configs/
│   ├── base.py
│   └── ld_resolution1024_test.py
├── models/
│   ├── custom_pipeline.py
│   ├── custom_model_mmdit.py
│   ├── custom_model_xvae.py
│   └── adapter.py
├── benchmark/
│   └── RevealLayerBench.json # 评测 JSON manifest
└── diffusers/                # 项目自带的 diffusers 源码，需要本地安装
```

真正运行推理的是 `infer.py`。`infer.sh` 可以作为参考，但不能直接照搬，因为它硬编码了作者机器上的 Python：

```bash
/home/jovyan/boomcheng-work-shcdt/zhaoshihao/anaconda/envs/flux/bin/python
```

建议直接用当前环境的 `python infer.py ...` 运行。

## 2. 推荐运行环境

README 写明作者测试环境为：

- Python 3.10
- CUDA GPU
- Linux / linux-64
- bfloat16 推理

建议使用 Linux 或 WSL2 + CUDA。原生 Windows 可能会卡在 `flash-attn`、`xformers`、CUDA 扩展或 bash 脚本兼容性上。

推荐创建环境：

```bash
conda create -n reveallayer python=3.10
conda activate reveallayer
```

安装依赖时注意：当前 `requirements.txt` 最后两行写成了：

```text
pip install peft==0.15.2
pip install xformers==0.0.31.post1
```

这不是标准 requirements 格式，直接 `pip install -r requirements.txt` 可能失败。推荐手动安装：

```bash
pip install opencv-python-headless==4.10.0.84
pip install torch==2.7.1 torchvision==0.22.1
pip install pillow==10.3.0 tqdm==4.65.0 mmengine==0.10.7
pip install safetensors==0.4.3 huggingface-hub==0.36.0
pip install einops==0.8.0 accelerate==1.6.0 transformers==4.57.1
pip install protobuf==3.20.2 sentencepiece==0.1.99
pip install peft==0.15.2
pip install xformers==0.0.31.post1
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

然后安装项目自带的 `diffusers`：

```bash
cd diffusers
pip install .
cd ..
```

运行前建议设置 `PYTHONPATH`：

```bash
export PYTHONPATH="$(pwd):$(pwd)/diffusers/src:${PYTHONPATH}"
```

## 3. 使用 Podman 构建 Docker 镜像

项目已提供 Dockerfile：

```text
docker/Dockerfile
```

这个镜像主要打包运行环境：CUDA、Python、PyTorch、xformers、flash-attn、项目自带的 diffusers。权重、数据和当前项目代码建议运行时挂载进去，不建议打进镜像。

镜像基于：

```text
docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
```

并设置了：

```text
TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
```

用于覆盖 A6000、L20、H20 这几类 GPU：

- A6000：Ampere，compute capability 8.6
- L20：Ada，compute capability 8.9
- H20：Hopper，compute capability 9.0

在项目根目录执行构建：

```bash
podman build -f docker/Dockerfile -t reveallayer:cuda128 .
```

如果构建 `flash-attn` 时机器 CPU/内存压力过大，可以调低并行编译数：

```bash
podman build -f docker/Dockerfile -t reveallayer:cuda128 --build-arg MAX_JOBS=4 .
```

如果已经构建过 `reveallayer:cuda128`，但运行时报：

```text
Requires Flash-Attention version >=2.7.1,<=2.8.0 but got 2.8.3.post1
```

可以不重建完整镜像，直接用增量 Dockerfile 修复：

```bash
podman build -f docker/Dockerfilev1.1 -t reveallayer:cuda128-v1.1 .
```

之后运行时把镜像名换成：

```bash
reveallayer:cuda128-v1.1
```

如果构建机无法直接访问 PyPI、PyTorch 或 Hugging Face，需要先配置 Podman/系统代理，或者在可联网环境构建后再迁移镜像。

构建后先验证容器是否能看到 GPU：

```bash
podman run --rm --device nvidia.com/gpu=all reveallayer:cuda128 \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果你的 Podman/NVIDIA Container Toolkit 版本不支持 `--device nvidia.com/gpu=all`，需要先在宿主机配置 NVIDIA CDI，或按当前机器的 Podman GPU 方案改成对应参数。

进入容器开发/推理：

```bash
podman run --rm -it \
  --device nvidia.com/gpu=all \
  -v "$PWD:/workspace/RevealLayer:Z" \
  -w /workspace/RevealLayer \
  reveallayer:cuda128 \
  bash
```

如果不是 SELinux 环境，或者在 WSL/Windows Podman 中挂载报错，可以去掉 `:Z`：

```bash
podman run --rm -it \
  --device nvidia.com/gpu=all \
  -v "$PWD:/workspace/RevealLayer" \
  -w /workspace/RevealLayer \
  reveallayer:cuda128 \
  bash
```

容器内单卡推理示例：

```bash
python infer.py \
  --input_json ./benchmark/RevealLayerBench.json \
  --output_dir ./results/RevealLayerV1_test \
  --pretrained_model_name_or_path ./models/FLUX.1-dev \
  --ckpt_dir ./models/RevealLayer \
  --transp_vae_ckpt ./models/RevealLayer/xvae/transparent_decoder_ckpt.pth \
  --cfg_path ./configs/ld_resolution1024_test.py \
  --gpu_id 0 \
  --max_samples 1 \
  --cfg 1.0 \
  --steps 30 \
  --tols 1 \
  --cid 0
```

注意：Dockerfile 不会下载 FLUX.1-dev、RevealLayer、ViT-B/32 或数据集。请按下面章节把权重和数据放到项目目录中，再通过上面的 `-v "$PWD:/workspace/RevealLayer"` 挂载进容器。

## 4. 需要下载的权重

需要至少三类权重：

1. RevealLayer 官方权重
2. FLUX.1-dev 基座模型
3. ViT-B/32 torchvision 权重

先启用 Git LFS：

```bash
git lfs install
```

### 4.1 RevealLayer 权重

README 给出的地址：

```bash
git clone https://huggingface.co/qihoo360/RevealLayer models/RevealLayer
```

建议放到：

```text
RevealLayer/models/RevealLayer/
```

推理代码会从 `--ckpt_dir` 中读取：

```text
models/RevealLayer/
├── pytorch_lora_weights.safetensors
├── layer_pe.pt
├── Refiner.pt
└── xvae/
    └── transparent_decoder_ckpt.pth
```

注意：`infer.py` 默认值是 `./ckpts/RevealLayer` 和 `./ckpts/xvae/transparent_decoder_ckpt.pth`，但 README 推荐下载到 `models/RevealLayer`。因此运行时建议显式传参：

```bash
--ckpt_dir ./models/RevealLayer
--transp_vae_ckpt ./models/RevealLayer/xvae/transparent_decoder_ckpt.pth
```

### 4.2 FLUX.1-dev

README 给出的地址：

```bash
git clone https://huggingface.co/black-forest-labs/FLUX.1-dev models/FLUX.1-dev
```

建议放到：

```text
RevealLayer/models/FLUX.1-dev/
```

该目录下应包含：

```text
models/FLUX.1-dev/
├── transformer/
├── vae/
├── text_encoder/
├── text_encoder_2/
├── tokenizer/
├── tokenizer_2/
└── ...
```

运行时传：

```bash
--pretrained_model_name_or_path ./models/FLUX.1-dev
```

FLUX.1-dev 可能需要 Hugging Face 登录并接受模型 license：

```bash
huggingface-cli login
```

### 4.3 ViT-B/32 权重

这是代码里额外需要的权重，不在 README 的模型目录说明里，但 `models/custom_model_xvae.py` 会加载它：

```text
vit_b_32-d86f8d99.pth
```

本地运行时下载到：

```text
RevealLayer/checkpoints/vit32/vit_b_32-d86f8d99.pth
```

当前代码已经会从项目目录下的 `checkpoints/vit32` 读取：

```python
weights_base_dir = os.path.join(project_root, "checkpoints", "vit32")
```

如果该文件不存在，运行到 `CustomVAE()` 初始化时会报找不到 ViT 权重。

## 5. 数据需要放到哪里

当前仓库只有：

```text
benchmark/RevealLayerBench.json
```

没有实际图片数据目录。这个 JSON 里的图片路径类似：

```json
"full_image": "./RevealLayerBench/00027/full_image.png"
```

因此评测数据应下载/解压到项目根目录下：

```text
RevealLayer/
└── RevealLayerBench/
    ├── 00027/
    │   ├── full_image.png
    │   ├── background.png
    │   └── layer_0.png
    ├── 00047/
    │   └── ...
    └── ...
```

也就是说，`benchmark/RevealLayerBench.json` 期望从项目根目录运行，并能找到：

```text
./RevealLayerBench/<imgid>/full_image.png
```

README 中提到数据集地址：

```text
https://huggingface.co/datasets/qihoo360/RevealLayer-100K
```

但 README 的 TODO 也写了 `RevealLayer-100K` 和 `RevealLayerBench` 仍待发布，所以如果 Hugging Face 上尚未提供完整 bench 图片，就需要使用自己的图片和 bbox 生成 JSON。

自定义输入 JSON 格式：

```json
[
  {
    "imgid": "example_001",
    "full_image": "./my_data/example_001/full_image.png",
    "detections": [
      {
        "bbox": [x1, y1, x2, y2]
      }
    ]
  }
]
```

`background` 和 `LayerInfoRaw` 只用于评测/记录，推理主逻辑主要使用：

- `imgid`
- `full_image`
- `detections[*].bbox`

当前 `infer.py` 支持通过 `--data_root` 指定数据集根目录。JSON 中的相对路径会和 `--data_root` 拼接；如果不传 `--data_root`，默认使用 `--input_json` 所在目录。

例如你的 JSON 在：

```text
E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200\metaData.json
```

并且 JSON 中的 `full_image` 是相对路径，那么可以这样运行：

```bash
python infer.py \
  --input_json "E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200\metaData.json" \
  --data_root "E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200" \
  ...
```

也可以省略 `--data_root`，代码会自动使用 `metaData.json` 所在文件夹作为数据根目录。

如果 JSON 中的路径本身已经从项目根目录开始写，例如仓库里的 `./RevealLayerBench/00027/full_image.png`，运行时应传：

```bash
--data_root .
```

## 6. 单卡推理命令

推荐先单卡跑 1 个样本验证环境：

```bash
python infer.py \
  --input_json ./benchmark/RevealLayerBench.json \
  --data_root . \
  --output_dir ./results/RevealLayerV1_test \
  --pretrained_model_name_or_path ./models/FLUX.1-dev \
  --ckpt_dir ./models/RevealLayer \
  --transp_vae_ckpt ./models/RevealLayer/xvae/transparent_decoder_ckpt.pth \
  --cfg_path ./configs/ld_resolution1024_test.py \
  --gpu_id 0 \
  --max_samples 1 \
  --cfg 1.0 \
  --steps 30 \
  --tols 1 \
  --cid 0
```

结果会保存到：

```text
results/RevealLayerV1_test/<imgid>/
├── original_full_image.png
├── resized_full_image.png
├── merged_image.png
├── fg_rgba.png
├── bg_rgba.png
├── layer_0_rgba.png
├── layer_0_rgba_mask.png
└── ...
```

## 7. 多 GPU 推理

`infer.py` 支持用 `--tols` 和 `--cid` 切分 JSON 样本：

- `--tols`：总切片数
- `--cid`：当前切片编号，从 `0` 到 `tols - 1`

例如 8 张 GPU 各跑一份：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py ... --gpu_id 0 --tols 8 --cid 0
CUDA_VISIBLE_DEVICES=1 python infer.py ... --gpu_id 0 --tols 8 --cid 1
CUDA_VISIBLE_DEVICES=2 python infer.py ... --gpu_id 0 --tols 8 --cid 2
CUDA_VISIBLE_DEVICES=3 python infer.py ... --gpu_id 0 --tols 8 --cid 3
CUDA_VISIBLE_DEVICES=4 python infer.py ... --gpu_id 0 --tols 8 --cid 4
CUDA_VISIBLE_DEVICES=5 python infer.py ... --gpu_id 0 --tols 8 --cid 5
CUDA_VISIBLE_DEVICES=6 python infer.py ... --gpu_id 0 --tols 8 --cid 6
CUDA_VISIBLE_DEVICES=7 python infer.py ... --gpu_id 0 --tols 8 --cid 7
```

这里 `CUDA_VISIBLE_DEVICES=3` 后，进程内部看到的可用 GPU 编号通常还是 `0`，所以 `--gpu_id 0` 更稳。

## 8. 容易踩坑的地方

1. `infer.py` 默认输入 JSON 是 `./benchmark/RevealLayerBenchMark-200.json`，但仓库实际文件是 `./benchmark/RevealLayerBench.json`，运行时必须显式传 `--input_json ./benchmark/RevealLayerBench.json`。
2. `infer.py` 默认 FLUX 路径是作者机器路径，必须传 `--pretrained_model_name_or_path ./models/FLUX.1-dev`。
3. `infer.py` 默认 RevealLayer 权重路径是 `./ckpts/RevealLayer`，但 README 推荐下载到 `models/RevealLayer`，必须统一路径或显式传 `--ckpt_dir`。
4. 透明 VAE 默认路径是 `./ckpts/xvae/transparent_decoder_ckpt.pth`，如果按 README 下载到 `models/RevealLayer/xvae/`，必须传 `--transp_vae_ckpt`。
5. `models/custom_model_xvae.py` 会从 `checkpoints/vit32/vit_b_32-d86f8d99.pth` 读取 ViT-B/32 权重，需要提前放好该文件。
6. `infer.sh` 硬编码了作者机器的 conda Python，建议不用它，或改成当前环境的 `python`。
7. `requirements.txt` 最后两行格式不标准，建议按本文手动安装依赖。
8. `benchmark/RevealLayerBench.json` 只是一份 manifest，实际图片目录 `RevealLayerBench/` 不在当前仓库中，需要额外下载或自行准备。

## 9. 推荐最终目录结构

```text
RevealLayer/
├── benchmark/
│   └── RevealLayerBench.json
├── RevealLayerBench/
│   ├── 00027/
│   │   ├── full_image.png
│   │   ├── background.png
│   │   └── layer_0.png
│   └── ...
├── models/
│   ├── RevealLayer/
│   │   ├── pytorch_lora_weights.safetensors
│   │   ├── layer_pe.pt
│   │   ├── Refiner.pt
│   │   └── xvae/
│   │       └── transparent_decoder_ckpt.pth
│   ├── FLUX.1-dev/
│   │   ├── transformer/
│   │   ├── vae/
│   │   ├── text_encoder/
│   │   ├── text_encoder_2/
│   │   ├── tokenizer/
│   │   └── tokenizer_2/
├── checkpoints/
│   └── vit32/
│       └── vit_b_32-d86f8d99.pth
├── results/
├── diffusers/
├── infer.py
└── Reveallayer.md
```
