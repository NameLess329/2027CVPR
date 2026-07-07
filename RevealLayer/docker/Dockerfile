# syntax=docker/dockerfile:1

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    PYTHONPATH=/workspace/RevealLayer:/workspace/RevealLayer/diffusers/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        libglib2.0-0 \
        libgl1 \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja

WORKDIR /workspace/RevealLayer

# Install PyTorch first so flash-attn can compile against the same CUDA/PyTorch ABI.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 \
        torchvision==0.22.1

RUN python -m pip install \
        opencv-python-headless==4.10.0.84 \
        pillow==10.3.0 \
        tqdm==4.65.0 \
        mmengine==0.10.7 \
        safetensors==0.4.3 \
        huggingface-hub==0.36.0 \
        einops==0.8.0 \
        accelerate==1.6.0 \
        transformers==4.57.1 \
        protobuf==3.20.2 \
        sentencepiece==0.1.99 \
        peft==0.15.2 \
        xformers==0.0.31.post1

# xformers==0.0.31.post1 requires Flash-Attention >=2.7.1 and <=2.8.0.
# PyPI does not publish an exact 2.8.0 release, and 2.8.0.post2 may compare as >2.8.0.
RUN python -m pip install flash-attn==2.7.4.post1 --no-build-isolation \
    && python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"

# Copy the local fork of diffusers and install it into the image.
COPY diffusers /tmp/reveallayer-diffusers
RUN python -m pip install /tmp/reveallayer-diffusers \
    && rm -rf /tmp/reveallayer-diffusers

CMD ["/bin/bash"]
