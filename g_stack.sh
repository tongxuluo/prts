#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job prts
#SBATCH --partition=Your partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=28
#SBATCH --time=7-00:00:00
#SBATCH --output=prts.out
#SBATCH --error=prts.err
#SBATCH --exclusive

cd /path/to/your/prts

TIME=$(date +%Y-%m-%d-%H-%M-%S)
SCRIPT_DIR=/aifs4su/data/tongxuluo/PRTS
MODEL_NAME=24L2048H
METHOD=stacking
CONFIG="./prts_configs/stacking_6L_24L.json"
export WANDB_API_KEY=1097fdf04de2e945cb795b86d6b2a52820e988d9

source /home/tongxuluo/env/anaconda3/bin/activate
conda activate tinyllama

srun python pretrain/run_pretrain.py \
    --num_nodes=4 \
    --model_name=${MODEL_NAME} \
    --name=${MODEL_NAME} \
    --method=${METHOD} \
    --config_path=${CONFIG} \
    --out_dir=${SCRIPT_DIR}/${METHOD}/${TIME} \
    --train_data_dir=/path/to/your/slimpajama \
    --devices=8 \
    --global_batch_size=1024 \
    --learning_rate=3e-4 \
    --min_lr=3e-5 \
    --micro_batch_size=8 \
    --max_step=300000 \
    --warmup_steps=3000 \
    --log_step_interval=1 \
    --eval_iters=10000 \
    --save_step_interval=5000 \
    --eval_step_interval=5000 \
    --weight_decay=1e-1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=True \
    --resume_id=5000