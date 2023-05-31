#!/bin/bash
SEEDS=(7)
LRS=(0.0005)
KLS=(0 0.1 1 10 100 1000)

for lr in ${LRS[@]}; do
    for seed in ${SEEDS[@]}; do
        for kl in ${KLS[@]}; do
            export WANDB_NAME=vip_seed_${seed}_lr_${lr}_kl_${kl}
            sbatch --job-name=cg_eg_vip run_vip.slurm ${seed} ${lr} ${kl}
        done
    done
done