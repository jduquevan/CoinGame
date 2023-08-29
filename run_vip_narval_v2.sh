#!/bin/bash
SEEDS=(8)
ALRS=(0.007 0.005)
CLRS=(0.01)
OPTS=("adam")
INFS=(0.3 0.4)
ENTS=(0.05 0.07)
RLLS=(20)

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