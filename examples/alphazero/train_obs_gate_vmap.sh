#!/bin/bash
#SBATCH --job-name=alphazero_obs_gate_jit_vmap                   # Default job name (sweep overrides)
#SBATCH -c 2                               # CPU cores per task
#SBATCH -t 0-17:10                         # Runtime (D-HH:MM)
#SBATCH -p kempner_h100                    # Partition
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:1                       # 1 GPU
#SBATCH --mem=80G                          # RAM for the job
#SBATCH -o slurm-%x-%j_alphazero_obs_gate_jit_vmap.out                 # STDOUT (%x=jobname, %j=jobid)
#SBATCH -e slurm-%x-%j_alphazero_obs_gate_jit_vmap.err                 # STDERR

# Load modules / env
module load python/3.10.9-fasrc01

# If you rely on conda commands:
# source ~/.bashrc || true
# conda activate torch || true
cd /n/home04/amuppidi/speedchess/examples/alphazero
# Use explicit Python path from your torch env (as in your example)
~/.conda/envs/torch/bin/python train_gate_speed_gardner_jit_vmap.py
