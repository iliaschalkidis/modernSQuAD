#!/bin/bash
#SBATCH --cpus-per-task=8 --mem=16000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=train_encoder_modern_bert.txt
#SBATCH --time=5:00:00

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate squad

echo $SLURMD_NODENAME

# Model Parameters
MODEL_PATH='answerdotai/ModernBERT-base'

export PYTHONPATH=.

python ./src/trainer/train_encoder.py \
  --model_name ${MODEL_PATH} \
  --learning_rate 5e-5 \
  --max_epochs 10