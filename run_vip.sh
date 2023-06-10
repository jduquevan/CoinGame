#!/bin/bash
SEEDS=(1)
LRS=(0.001)
KLS=(0)

for lr in ${LRS[@]}; do
    for seed in ${SEEDS[@]}; do
        for kl in ${KLS[@]}; do
            export WANDB_NAME=vip_seed_${seed}_lr_${lr}_kl_${kl}
            sbatch --job-name=cg_eg_vip run_vip.slurm ${seed} ${lr} ${kl} --exclude=cn-g023
        done
    done
done