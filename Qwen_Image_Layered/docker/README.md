# Podman image for Qwen-Image-Layered

Build from the repository root:

```bash
podman build -f docker/Containerfile -t qwen-image-layered:cu124 .
```

Run with NVIDIA GPUs:

```bash
podman run --rm -it --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$PWD:/workspace/Qwen_Image_Layered" \
  qwen-image-layered:cu124
```

Inside the container:

```bash
python scripts/Qwen-Image-Layered.py
bash scripts/Qwen-Image-Layered.sh
```

For older Podman/NVIDIA setups, replace `--device nvidia.com/gpu=all` with the GPU option supported by that host, such as `--gpus all` if available.
