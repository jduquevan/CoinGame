#!/bin/bash
SEEDS=(1)
LRS=(0.005)
GPS=(0.1 0.9)
ENTS=(0.0005 0.001 0.005 0.01 0.05)

for lr in ${LRS[@]}; do
    for seed in ${SEEDS[@]}; do
        for gp in ${GPS[@]}; do
            for ent in ${ENTS[@]}; do
                export WANDB_NAME=vip_seed_${seed}_lr_${lr}_gp_${gp}_ent_${ent}
                sbatch --job-name=cg_eg_vip run_vip_narval.slurm ${seed} ${lr} ${gp} ${ent}
            done
        done
    done
done