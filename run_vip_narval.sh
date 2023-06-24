#!/bin/bash
SEEDS=(1)
LRS=(0.005)
GPS=(0.7)
ENTS=(0.01)
KLS=(0.005)
HLENS=(8)

for lr in ${LRS[@]}; do
    for seed in ${SEEDS[@]}; do
        for gp in ${GPS[@]}; do
            for ent in ${ENTS[@]}; do
                for kl in ${KLS[@]}; do
                    for hlen in ${HLENS[@]}; do
                        export WANDB_NAME=vip_seed_${seed}_lr_${lr}_gp_${gp}_ent_${ent}_kl_${kl}_hlen_${hlen}
                        sbatch --job-name=cg_eg_vip run_vip_narval.slurm ${seed} ${lr} ${gp} ${ent} ${kl} ${hlen}
                    done
                done
            done
        done
    done
done