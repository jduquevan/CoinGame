#!/bin/bash
SEEDS=(8)
ALRS=(0.0005 0.001)
CLRS=(0.01)
OPTS=("adam" "om")
INFS=(1)
ENTS=(0.15 0.2 0.25)
RLLS=(25)

for alr in ${ALRS[@]}; do
    for clr in ${CLRS[@]}; do
        for opt in ${OPTS[@]}; do
            for inf in ${INFS[@]}; do
                for ent in ${ENTS[@]}; do
                    for rll in ${RLLS[@]}; do
                        for seed in ${SEEDS[@]}; do
                            export WANDB_NAME=vip_v2_seed_${seed}_alr_${alr}_clr_${clr}_opt_${opt}_inf_${inf}_ent_${ent}_rll_${rll}
                            sbatch --job-name=ipd_vip run_vip_narval_v2.slurm ${seed} ${alr} ${clr} ${opt} ${inf} ${ent} ${rll}
                        done           
                    done
                done
            done
        done
    done
done