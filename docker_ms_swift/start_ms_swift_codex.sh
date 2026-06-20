#!/usr/bin/env bash
set -euo pipefail

if docker ps -a --format '{{.Names}}' | grep -Fxq ms_swift_codex_root; then
  docker rm -f ms_swift_codex_root
fi

docker run --gpus all -it \
  --name ms_swift_codex_root \
  --ipc=host \
  --shm-size=64g \
  --network host \
  --user root \
  -v "${MS_SWIFT_WORKSPACE:-$HOME/ms_swift_codex_work}:/workspace" \
  -v "${CODEX_HOME:-$HOME/.codex}:/root/.codex" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/cache/huggingface \
  -e HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets \
  -e MODELSCOPE_CACHE=/workspace/cache/modelscope \
  -e TORCH_HOME=/workspace/cache/torch \
  -e WANDB_DIR=/workspace/cache/wandb \
  -w /workspace \
  ms-swift-codex-root:cu121
