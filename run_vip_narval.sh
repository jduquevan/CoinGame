#!/bin/bash
SEEDS=(7)
LRS=(0.0005)
GPS=(0.5 0.6 0.7)
ENTS=(0.015 0.05)
KLS=(0.05)
HLENS=(8)
RDPS=(0 0.05 0.15 0.25)

for lr in ${LRS[@]}; do
    for seed in ${SEEDS[@]}; do
        for gp in ${GPS[@]}; do
            for ent in ${ENTS[@]}; do
                for kl in ${KLS[@]}; do
                    for hlen in ${HLENS[@]}; do
                        for rdp in ${RDPS[@]}; do
                            export WANDB_NAME=vip_seed_${seed}_lr_${lr}_gp_${gp}_ent_${ent}_kl_${kl}_hlen_${hlen}_rdp_${rdp}
                            sbatch --job-name=cg_eg_vip run_vip_narval.slurm ${seed} ${lr} ${gp} ${ent} ${kl} ${hlen} ${rdp}
                        done
                    done
                done
            done
        done
    done
done