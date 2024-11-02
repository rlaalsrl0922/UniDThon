#!/bin/bash

#SBATCH -J FacerAin
#SBATCH --nodelist=moana-y2
#SBATCH --partition batch_ce_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o logs/%A-%x.out
#SBATCH -e logs/%A-%x.err
#SBATCH --time=1-0

python3 main.py

# letting slurm know this code finished without any problem
exit 0
