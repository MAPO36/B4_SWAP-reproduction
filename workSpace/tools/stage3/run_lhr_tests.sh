#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/ubuntu/B4_2025/SWAP/WMP
export PYTHONPATH=$PWD
export MPLBACKEND=Agg
python /tmp/test_lhr_terrain.py
