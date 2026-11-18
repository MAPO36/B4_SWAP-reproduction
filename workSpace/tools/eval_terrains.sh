#!/usr/bin/env bash
# 三种地形自动评估（无需键盘）：slope / stair / gap，每种地形多个随机种子取平均
# 用法：bash eval_terrains.sh [run_name]        （默认 go2_long2000）
#       SEEDS="1 2 3" bash eval_terrains.sh ...  （默认就是 1 2 3）
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${1:-go2_long2000}"
SEEDS="${SEEDS:-1 2 3}"
cd "$ROOT/WMP" || exit 1

ln -sfn "$RUN" logs/go2_amp_example/WMP
echo "run=$RUN  ckpt=$(ls -1 logs/go2_amp_example/WMP/model_*.pt | sort -t_ -k2 -n | tail -1 | xargs basename)"
echo ""

printf "%-8s %-10s %-10s %-8s\n" "地形" "seed" "reward" "episodes"
printf "%s\n" "--------------------------------------------"

for t in slope stair gap; do
  sum=0; n=0; ep_sum=0
  for s in $SEEDS; do
    log="/tmp/eval_${t}_${s}.log"
    PYTHONPATH="$PWD" timeout 900 python legged_gym/scripts/play.py \
      --task=go2_amp --headless --sim_device=cuda:0 --terrain="$t" --seed="$s" > "$log" 2>&1
    reward=$(grep -a "total reward" "$log" | tail -1 | grep -oE "[-0-9.]+" | head -1)
    eps=$(grep -a "Total number of episodes" "$log" | tail -1 | awk '{print $NF}')
    printf "%-8s %-10s %-10s %-8s\n" "$t" "$s" "${reward:-NA}" "${eps:-NA}"
    if [ -n "${reward:-}" ]; then sum=$(awk -v a="$sum" -v b="$reward" 'BEGIN{print a+b}'); n=$((n+1)); fi
    if [ -n "${eps:-}" ]; then ep_sum=$((ep_sum+eps)); fi
    cp -f "$log" "$ROOT/workSpace/runs/stage-0/eval_${t}_seed${s}.log" 2>/dev/null || true
  done
  if [ "$n" -gt 0 ]; then
    printf "%-8s %-10s %-10s %-8s\n" "$t" "平均" \
      "$(awk -v s="$sum" -v n="$n" 'BEGIN{printf "%.3f", s/n}')" \
      "$(awk -v s="$ep_sum" -v n="$n" 'BEGIN{printf "%.1f", s/n}')"
  fi
  printf "%s\n" "--------------------------------------------"
done

echo ""
echo "指标：reward 越大越好；episodes = episode 结束次数，越少说明越少摔倒/重置"
