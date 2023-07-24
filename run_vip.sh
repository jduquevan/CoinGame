#!/bin/bash
SEEDS=(8)
LRS=(0.0005)
GPS=(0)
ENTS=(0)
KLS=(0.01)

for lr in ${LRS[@]}; do
    for seed in ${SEEDS[@]}; do
        for gp in ${GPS[@]}; do
            for ent in ${ENTS[@]}; do
                for kl in ${KLS[@]}; do
                    export WANDB_NAME=vip_seed_${seed}_lr_${lr}_gp_${gp}_ent_${ent}_kl_${kl}
                    sbatch --job-name=cg_eg_vip run_vip.slurm ${seed} ${lr} ${gp} ${ent} ${kl} 
                done
            done
        done
    done
done
