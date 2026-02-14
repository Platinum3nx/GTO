#!/bin/bash
#SBATCH --job-name=gto_train           # Job name
#SBATCH --output=runs/slurm-%j.out     # Standard output log
#SBATCH --error=runs/slurm-%j.err      # Standard error log
#SBATCH --partition=gpu                # Partition
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --constraint="a100_40gb|a100_80gb" # Request A100 (40GB or 80GB)
#SBATCH --cpus-per-task=8              # 8 CPU cores
#SBATCH --mem=64000                    # 64GB System RAM (in MB)
#SBATCH --time=12:00:00                # Time limit (12 hours)
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=qat5sc@virginia.edu # User email (replace with your computing ID)

# 1. Load Modules
module purge
module load gcc
module load python
module load cuda

# 2. Environment Setup
# Enhance performance for PyTorch on A100
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# Create/Activate Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip
    pip install -e .[dev]
else
    source venv/bin/activate
fi

# 3. Validation
echo "Job running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Request: A100"
nvidia-smi

# 4. Run Training
# Hyperparameters optimized for A100 (40GB+)
# - batch_size: 4096 (fits comfortably in VRAM)
# - parallel_games: 1024 (maximizes vectorized throughput)
# - precision: bf16 (native support on Ampere)

python -m gto.train.trainer \
  --subgame_board Kh8s2c \
  --learning_rate 3e-4 \
  --batch_size 4096 \
  --parallel_games 1024 \
  --trajectories_per_iter 1024 \
  --iterations 4000 \
  --checkpoint_dir checkpoints \
  --precision bf16 \
  --device cuda
