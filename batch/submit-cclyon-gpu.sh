#!/bin/bash
# Submits a CaloINN training job to CClyon GPU partition (8h, 1× V100).
#
# Usage:
#   sbatch batch/submit_cclyon_8h_1GPU.sh
#   sbatch batch/submit_cclyon_8h_1GPU.sh --config params/lemurs_fcceeallegro.yaml

#SBATCH --job-name=toynce
#SBATCH --output=slurm_logs/cclyon-%j-%x.out
#SBATCH --error=slurm_logs/cclyon-%j-%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@240
#SBATCH --partition=gpu_v100

$@
