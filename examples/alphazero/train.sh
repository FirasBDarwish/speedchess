#!/bin/bash
#SBATCH --job-name=alphazero_ckpt                   # Default job name (sweep overrides)
#SBATCH -c 2                               # CPU cores per task
#SBATCH -t 0-07:10                         # Runtime (D-HH:MM)
#SBATCH -p seas_gpu                    # Partition
#SBATCH --account=gershman_lab
#SBATCH --gres=gpu:4                       # 1 GPU
#SBATCH --mem=80G                          # RAM for the job
#SBATCH -o slurm-%x-%j_alphazero_ckpt.out                 # STDOUT (%x=jobname, %j=jobid)
#SBATCH -e slurm-%x-%j_alphazero_ckpt.err                 # STDERR

# Load modules / env
module load python/3.10.9-fasrc01

# If you rely on conda commands:
# source ~/.bashrc || true
# conda activate torch || true
cd /n/home04/amuppidi/speedchess/examples/alphazero
# Use explicit Python path from your torch env (as in your example)
~/.conda/envs/torch/bin/python train.py --num_simulations 2
