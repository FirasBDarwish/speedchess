#!/bin/bash

#SBATCH -J az_simbudget              # Job name
#SBATCH -o az_simbudget.out          # Stdout
#SBATCH -e az_simbudget.err          # Stderr
#SBATCH -t 48:00:00                  # Max runtime
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:4
#SBATCH -n 1                         # One MPI task
#SBATCH -N 1                         # One node
#SBATCH -c 6                         # 6 CPUs
#SBATCH --mem=128G                   # Memory

###############################################################
# Setup environment
###############################################################
module purge

# Activate conda environment
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate speedchess

echo ">>> Environment Info:"
conda info | grep "active environment"
which python
python - << EOF
import torch
print("Python:", __import__("sys").executable)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
EOF

echo "=== DIAGNOSTICS START ==="
echo "Host:" $(hostname)
echo "Date:" $(date)

echo -e "\n--- GPU Info (nvidia-smi) ---"
nvidia-smi

echo -e "\n--- CUDA Compiler (nvcc) ---"
# Checks system-level CUDA, helpful to see if it mismatches with conda
nvcc --version || echo "nvcc not found"

echo -e "\n--- Environment Variables ---"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo -e "\n--- Python Environment ---"
which python
python --version

echo -e "\n--- Installed Key Packages ---"
# Checks versions of torch and pydantic
pip list | grep -E "torch|pydantic|numpy|wandb"

echo -e "\n--- PyTorch Internal Check ---"
# This is the most important check. It asks Python directly if it sees the GPU.
python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version in Torch: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else \"None\"}')"

python -c "import jax; print(f'JAX Devices: {jax.devices()}')"

echo "=== DIAGNOSTICS END ==="
echo ""

###############################################################
# Simulation budgets to test
###############################################################
SIMS=(2 4 8 16 32)
export WANDB_API_KEY="75c36097080b86af5ad31abf5694aa4932c5431b"

# Completely disable JAX/XLA autotuning.
# This fixes "No valid config found" by skipping the benchmark step entirely.
export XLA_FLAGS="--xla_gpu_autotune_level=0"

###############################################################
# Main loop: run training for each simulation count
###############################################################
for SIM in "${SIMS[@]}"; do
    echo "===================================================="
    echo " Running AlphaZero training with num_simulations = ${SIM}"
    echo "===================================================="

    # Run your training script with overridden config
    python -m train \
        num_simulations=$SIM \
        wandb.group="sim_${SIM}" \
        run_name="sim_${SIM}" \
        +save_suffix="_sim${SIM}"
done
