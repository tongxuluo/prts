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
SCRIPT_DIR=/path/to/your/prts
MODEL_NAME=6L2048H
METHOD=scratch
export WANDB_API_KEY=Your WANDB_API_KEY

source /path/to/your/anaconda3/bin/activate
conda activate your_env

srun python pretrain/run_pretrain.py \
    --num_nodes=4 \
    --model_name=${MODEL_NAME} \
    --name=${MODEL_NAME} \
    --method=${METHOD} \
    --out_dir=${SCRIPT_DIR}/${METHOD}/${TIME} \
    --train_data_dir=/path/to/your/slimpajama \
    --devices=8 \
    --global_batch_size=1024 \
    --learning_rate=1e-3 \
    --min_lr=1e-4 \
    --micro_batch_size=32 \
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
    --decay_lr=True