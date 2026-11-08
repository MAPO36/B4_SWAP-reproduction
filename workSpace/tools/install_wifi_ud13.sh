#!/usr/bin/env bash
# 水星 UD13（2357:0137, Realtek 802.11ac AC1300）驱动安装
# 需要 root：sudo bash install_wifi_ud13.sh
set -uo pipefail

VIDPID="2357 0137"
SYS=/sys/bus/usb/drivers

echo "===== 0. 环境检查 ====="
[ "$(id -u)" -eq 0 ] || { echo "必须用 sudo 运行"; exit 1; }
uname -r; lsusb -d 2357:0137 || { echo "设备没插好，找不到 2357:0137"; exit 1; }
mokutil --sb-state 2>/dev/null || true

echo ""
echo "===== 1. 先试内核自带 rtw88_8822bu（不下载任何东西）====="
modprobe rtw88_8822bu 2>&1 && echo "modprobe rtw88_8822bu: OK" || echo "modprobe 失败"
if [ -w "$SYS/rtw88_8822bu/new_id" ]; then
  echo "$VIDPID" > "$SYS/rtw88_8822bu/new_id" 2>/dev/null \
    && echo "已把 $VIDPID 强制绑到 rtw88_8822bu" \
    || echo "new_id 写入失败（说明内核驱动的 ID 表不认这颗芯片）"
fi
sleep 3
echo "--- dmesg ---"
dmesg | tail -12
echo "--- 网卡接口 ---"
ip -br link | grep -vE "^lo|docker|br-|veth|tailscale"

if ip -br link | grep -qE "^wlx|^wlan[1-9]"; then
  echo ""
  echo ">>> 内核驱动直接可用，不需要装外置驱动。"
  exit 0
fi

echo ""
echo "===== 2. 内核驱动不行，编译外置 88x2bu 驱动 ====="
apt-get install -y build-essential dkms git bc >/dev/null 2>&1 || true
mkdir -p /root/src && cd /root/src || exit 1

if [ -d 88x2bu-20210702 ]; then
  cd 88x2bu-20210702 && git pull --ff-only 2>/dev/null
else
  echo "从 GitHub 克隆（慢的话会自动换镜像）..."
  if ! timeout 180 git clone --depth 1 https://github.com/morrownr/88x2bu-20210702.git 2>/dev/null; then
    echo "GitHub 失败，改用 ghproxy 镜像"
    rm -rf 88x2bu-20210702
    timeout 180 git clone --depth 1 https://ghproxy.net/https://github.com/morrownr/88x2bu-20210702.git \
      || { echo "镜像也失败，请手动下载后放到 /root/src/"; exit 1; }
  fi
  cd 88x2bu-20210702 || exit 1
fi

echo "开始构建安装（约 2-3 分钟）..."
./install-driver.sh NoPrompt 2>&1 | tail -25

echo ""
echo "===== 3. 结果 ====="
dkms status | grep -i 88x2bu || echo "dkms 里没有 88x2bu"
lsmod | grep -i 88x2bu || echo "模块未加载"
ip -br link | grep -vE "^lo|docker|br-|veth|tailscale"
echo ""
echo "如果看到 wlx... 接口，重启后即可在 GNOME 网络设置里连接 Wi-Fi。"
