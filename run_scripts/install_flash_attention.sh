#!/bin/bash
#SBATCH --cpus-per-task=8 --mem=16000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=flash_att.txt
#SBATCH --time=0:30:00

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate squad

MAX_JOBS=4 pip install flash-attn --no-build-isolation
