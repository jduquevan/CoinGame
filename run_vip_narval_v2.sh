#!/bin/bash
SEEDS=(8)
ALRS=(0.00001)
CLRS=(0.001)
OPTS=("eg")
INFS=(0.9)
ENTS=(0.1 0.2 0.3)

for alr in ${ALRS[@]}; do
    for clr in ${CLRS[@]}; do
        for opt in ${OPTS[@]}; do
            for inf in ${INFS[@]}; do
                for ent in ${ENTS[@]}; do
                    for seed in ${SEEDS[@]}; do
                        export WANDB_NAME=vip_v2_mask_not_-A_R_seed_${seed}_alr_${alr}_clr_${clr}_opt_${opt}_inf_${inf}_ent_${ent}
                        sbatch --job-name=ipd_vip run_vip_narval_v2.slurm ${seed} ${alr} ${clr} ${opt} ${inf} ${ent}
                    done           
                done
            done
        done
    done
done