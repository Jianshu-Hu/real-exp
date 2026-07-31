FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# 替换清华源并安装基础依赖 (保留了必要的 USB 和 CycloneDDS)
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    python3-pip python3-venv python3-colcon-common-extensions \
    libusb-1.0-0-dev \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosbridge-suite \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    # 【修改说明】：在上面的列表中新增了 ros-humble-rosbridge-suite。
    # 它是 ROS 2 官方提供的 WebSocket 服务端，允许非 ROS 环境（如纯 Python 3.12 环境）通过网络与 ROS 2 通信。

# 安装 uv (你提供的 uv-installer.sh 就是这里执行的)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 配置 ROS 2 环境变量，并设置默认使用更稳定的 CycloneDDS
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc

WORKDIR /workspace