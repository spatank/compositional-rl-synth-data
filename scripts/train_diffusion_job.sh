#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=slurm/%A_%a_train_diffusion.out
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/spatank/compositional-rl-synth-data/venv/bin/activate

python /home/spatank/compositional-rl-synth-data/scripts/train_diffusion.py \
    --base_data_path /mnt/kostas-graid/datasets/spatank \
    --base_results_folder /home/spatank/compositional-rl-synth-data/results/diffusion \
    --gin_config_files /home/spatank/compositional-rl-synth-data/config/diffusion.gin \
    --dataset_type expert \
    --experiment_type smallscale \
    --element IIWA \
    --num_train 4 \
    --seed 42
