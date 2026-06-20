FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV MODELSCOPE_CACHE=/root/.cache/modelscope
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git git-lfs curl wget vim nano tmux htop \
    python3.10 python3.10-venv python3-pip \
    build-essential ninja-build cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg ripgrep unzip zip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && python -m pip install -U pip setuptools wheel

# PyTorch CUDA 12.1
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Common LLM / VLM dependencies
RUN pip install -U \
    transformers accelerate peft datasets evaluate \
    modelscope sentencepiece protobuf einops timm pillow \
    qwen-vl-utils deepspeed tensorboard wandb \
    opencv-python-headless

# ms-swift from source
RUN git clone https://github.com/modelscope/ms-swift.git /opt/ms-swift \
 && cd /opt/ms-swift \
 && pip install -e '.[all]'

WORKDIR /workspace

CMD ["/bin/bash"]
