# Podman build example:
#   podman build -f vllm_qwen/docker/Dockerfile -t qwen36-vllm:latest .
#
# Podman run example (requires NVIDIA Container Toolkit / CDI on the host):
#   podman run --rm -it \
#     --security-opt=label=disable \
#     --device nvidia.com/gpu=all \
#     --ipc=host \
#     -v /data/models:/models:Z \
#     -v /data/sa1b:/data/sa1b:Z \
#     -v /data/output:/data/output:Z \
#     qwen36-vllm:latest
#
# Notes:
# - vLLM official images already bundle a matching CUDA / PyTorch / vLLM stack.
# - A6000, L20, and H20 all work with this image family as long as the host
#   driver + NVIDIA Container Toolkit are configured correctly.

ARG VLLM_BASE_IMAGE=docker.io/vllm/vllm-openai:latest
FROM ${VLLM_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    MODELSCOPE_CACHE=/workspace/.cache/modelscope \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN command -v python >/dev/null 2>&1 || ln -s "$(command -v python3)" /usr/local/bin/python

COPY vllm_qwen/docker/requirements.docker.txt /tmp/requirements.docker.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /tmp/requirements.docker.txt

WORKDIR /workspace/app

# COPY vllm_qwen /workspace/app/vllm_qwen
# COPY download_ms.py /workspace/app/download_ms.py

# RUN mkdir -p \
#     /workspace/.cache/huggingface \
#     /workspace/.cache/modelscope \
#     /workspace/data \
    # /workspace/output

CMD ["/bin/bash"]
