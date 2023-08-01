#!/bin/bash
SEEDS=(8)
ALRS=(0.0005 0.001 0.005)
CLRS=(0.0005 0.001 0.005)
OPTS=("eg" "adam" "om")
GNS=(0.3)

for alr in ${ALRS[@]}; do
    for clr in ${CLRS[@]}; do
        for opt in ${OPTS[@]}; do
            for gn in ${GNS[@]}; do
                for seed in ${SEEDS[@]}; do
                    export WANDB_NAME=vip_seed_${seed}_alr_${alr}_clr_${clr}_opt_${opt}_gn_${gn}
                    sbatch --job-name=ipd_vip run_vip_narval.slurm ${seed} ${alr} ${clr} ${opt} ${gn}            
                done
            done
        done
    done
done