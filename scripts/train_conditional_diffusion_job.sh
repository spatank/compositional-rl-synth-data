#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=slurm/%A_%a_train_conditional_diffusion.out
#SBATCH --mem=320G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=96:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/spatank/compositional-rl-synth-data/venv/bin/activate

python /home/spatank/compositional-rl-synth-data/scripts/train_conditional_diffusion.py \
    --base_data_path /mnt/kostas-graid/datasets/spatank \
    --base_results_folder /mnt/kostas-graid/datasets/spatank/results/diffusion \
    --gin_config_files /home/spatank/compositional-rl-synth-data/config/diffusion.gin \
    --dataset_type expert \
    --experiment_type default \
    --num_train 1 \
    --seed 42
    