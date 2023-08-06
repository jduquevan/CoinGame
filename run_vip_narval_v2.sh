#!/bin/bash
SEEDS=(8)
ALRS=(0.000001 0.00001 0.0001)
CLRS=(0.0005 0.001 0.005)
OPTS=("eg" "adam" "om")
INFS=(1)
ENTS=(0.1)

for alr in ${ALRS[@]}; do
    for clr in ${CLRS[@]}; do
        for opt in ${OPTS[@]}; do
            for inf in ${INFS[@]}; do
                for ent in ${ENTS[@]}; do
                    for seed in ${SEEDS[@]}; do
                        export WANDB_NAME=vip_v2_seed_${seed}_alr_${alr}_clr_${clr}_opt_${opt}_inf_${inf}_ent_${ent}
                        sbatch --job-name=ipd_vip run_vip_narval_v2.slurm ${seed} ${alr} ${clr} ${opt} ${inf} ${ent}
                    done           
                done
            done
        done
    done
done