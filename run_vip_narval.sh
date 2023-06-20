#!/bin/bash
SEEDS=(1)
LRS=(0.005)
GPS=(0.1 0.9)
ENTS=(0.01)
KLS=(0.001 0.005 0.01 0.05 0.1)

for lr in ${LRS[@]}; do
    for seed in ${SEEDS[@]}; do
        for gp in ${GPS[@]}; do
            for ent in ${ENTS[@]}; do
                for kl in ${KLS[@]}; do
                    export WANDB_NAME=vip_seed_${seed}_lr_${lr}_gp_${gp}_ent_${ent}_kl_${kl}
                    sbatch --job-name=cg_eg_vip run_vip_narval.slurm ${seed} ${lr} ${gp} ${ent} ${kl}
                done
            done
        done
    done
done