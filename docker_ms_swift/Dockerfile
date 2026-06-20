FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget vim nano tmux htop \
    python3.10 python3.10-venv python3-pip \
    build-essential ninja-build cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg ripgrep unzip zip ca-certificates \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

RUN python -m pip install --no-cache-dir \
    transformers accelerate peft datasets evaluate \
    modelscope sentencepiece protobuf einops timm pillow \
    qwen-vl-utils deepspeed tensorboard wandb \
    opencv-python-headless

RUN git lfs install \
    && git clone https://github.com/modelscope/ms-swift.git /opt/ms-swift

WORKDIR /opt/ms-swift
RUN python -m pip install --no-cache-dir -e '.[all]'

RUN mv /usr/local/bin/swift /usr/local/bin/swift.real \
    && printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'if [ "$#" -eq 0 ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then' \
    '  cat <<'\''EOF'\''' \
    'Usage: swift <command> [args]' \
    '' \
    'Commands:' \
    '  pt' \
    '  sft' \
    '  infer' \
    '  merge-lora' \
    '  web-ui' \
    '  deploy' \
    '  rollout' \
    '  rlhf' \
    '  sample' \
    '  export' \
    '  eval' \
    '  app' \
    'EOF' \
    '  exit 0' \
    'fi' \
    'exec /usr/local/bin/swift.real "$@"' \
    > /usr/local/bin/swift \
    && chmod +x /usr/local/bin/swift

RUN npm install -g n \
    && n 20.19.0 \
    && hash -r \
    && npm install -g @openai/codex

ENV HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/huggingface \
    HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets \
    MODELSCOPE_CACHE=/workspace/cache/modelscope \
    TORCH_HOME=/workspace/cache/torch \
    WANDB_DIR=/workspace/cache/wandb

WORKDIR /workspace

CMD ["/bin/bash"]
