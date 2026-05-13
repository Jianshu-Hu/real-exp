# 严格对齐系统与 CUDA
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 将 Ubuntu 的 apt 源替换为清华源加速
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装 Python 3.12 和官方要求的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    build-essential git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev \
    cmake pkg-config ninja-build \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 将 Python 3.12 设置为系统的默认 python 命令
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# 将 uv 添加到环境变量中
ENV PATH="/root/.local/bin:$PATH"

# 设置 uv 默认使用清华 PyPI 镜像源
ENV UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

WORKDIR /workspace