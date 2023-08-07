#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=ipd_vip

# Remove one # to uncommment
#SBATCH --output=/scratch/jduque/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/jduque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-01:59:00   
#SBATCH --gres=gpu:1 

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=juan.duque@mila.quebec


# Submit jobs.
version=4
export WANDB_ENTITY="jduque"

module purge
eval "$(conda shell.bash hook)"
source /home/jduque/environments/cogames/bin/activate
module load cuda/11
export WANDB_MODE=offline

python -m src.run  env_type="ipd" seed=${1} vip_agent_ipd_v2.optim_config.lr=${2} \
    vip_agent_ipd_v2.critic_optim_config.lr=${3} base_agent.opt_type=${4}\
    vip_agent_ipd_v2.inf_weight=${5} vip_agent_ipd_v2.entropy_weight=${6} \
    hydra.run.dir="/scratch/jduque/hydra" wandb_dir="/scratch/jduque/wandb"

#python -m src.run seed=1 vip_agent.optim_config.lr=0.005 hydra.run.dir="/scratch/jduque/hydra" wandb_dir="/scratch/jduque/wandb"