#!/bin/bash
#SBATCH --job-name=graph            # Job name
#SBATCH --output=logs/job_%j.log    # Standard output/error log (%j = JobID)
#SBATCH --partition=main            # Partition/Queue name (e.g., short, gpu) [1, 3]
#SBATCH --gres=gpu:b200_1g.23gb:1   # Request 1 slice of the 23GB instance
#SBATCH --mem=16G                   # Job memory request (e.g., 16G, 100M) [14]
#SBATCH --time=02:00:00             # Time limit (HH:MM:SS) [12]


conda init
conda activate graph


export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep MIG | awk '{print $NF}' | tr -d ')' | head -n 1)
echo "Using MIG Device: $CUDA_VISIBLE_DEVICES"

HYDRA_FULL_ERROR=1 python main.py

