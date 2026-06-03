#!/bin/bash

echo "开始配置 GELLO 机械臂 USB 绑定规则..."

# 检查是否使用了 sudo
if [ "$EUID" -ne 0 ]; then
  echo "请使用 sudo 运行此脚本: sudo ./scripts/setup_usb_rules.sh"
  exit
fi

# 复制规则文件到系统目录
cp ./scripts/udev_rules/99-gello-usb.rules /etc/udev/rules.d/

# 重新加载 udev 规则
udevadm control --reload-rules
udevadm trigger

echo "USB 规则配置完成！"
echo "请确保右臂插在物理端口 3-2，左臂插在物理端口 3-3。"
echo "你可以使用 'ls -l /dev/ttyUSB*' 来检查映射是否成功。"
