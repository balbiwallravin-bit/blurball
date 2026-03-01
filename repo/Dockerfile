FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git ffmpeg \
    libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# ONNXRuntime GPU + 基础科学计算/图像处理
RUN pip install \
    numpy tqdm pandas \
    opencv-python-headless \
    onnx onnxruntime-gpu \
    pyyaml rich matplotlib

WORKDIR /workspace
