FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# ONNXRuntime GPU + 常用依赖
RUN pip install \
    numpy tqdm pandas pyyaml rich matplotlib \
    opencv-python-headless \
    onnx onnxruntime-gpu

# PyTorch（可选，但装上不亏，后面学生模型训练也要）
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
