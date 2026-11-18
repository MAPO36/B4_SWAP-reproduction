#!/usr/bin/env bash
# 把 2357:0137 手动绑定到外置 88x2bu 驱动
set -uo pipefail
[ "$(id -u)" -eq 0 ] || { echo "必须 sudo 运行"; exit 1; }

echo "===== 1. 卸载内核自带的 rtw88_8822bu（避免抢设备）====="
modprobe -r rtw88_8822bu 2>/dev/null && echo "已卸载" || echo "未加载或正在使用，跳过"

echo "===== 2. 加载外置 88x2bu ====="
modprobe 88x2bu 2>&1 && echo "加载成功" || { echo "加载失败，见 dmesg"; exit 1; }
lsmod | grep 88x2bu

echo "===== 3. 认领 2357:0137 ====="
if [ -e /sys/bus/usb/drivers/88x2bu/new_id ]; then
  echo "2357 0137" > /sys/bus/usb/drivers/88x2bu/new_id 2>/dev/null \
    && echo "new_id 写入成功" \
    || echo "new_id 写入失败（芯片可能不兼容该驱动）"
else
  echo "驱动没有 new_id 接口"
fi
sleep 4

echo "===== 4. 结果 ====="
echo "--- 接口 ---"
ip -br link | grep -vE "^lo|docker|br-|veth|tailscale"
echo "--- 内核日志 ---"
dmesg | tail -20 | grep -iE "88x2bu|rtw|wlan|usb 3-3|error|fail" || dmesg | tail -8

echo ""
if ip -br link | grep -qE "^wlx|^wlan[1-9]"; then
  echo ">>> 成功：新无线接口已出现"
else
  echo ">>> 仍无新接口，需要改驱动源码的 ID 表（下一步）"
fi
