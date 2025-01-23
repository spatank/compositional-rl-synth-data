#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=slurm/%A_%a_generate_samples.out
#SBATCH --mem=160G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/spatank/compositional-rl-synth-data/venv/bin/activate

python /home/spatank/compositional-rl-synth-data/scripts/generate_synthetic_data.py \
    --base_data_path /mnt/kostas-graid/datasets/spatank \
    --base_results_folder /mnt/kostas-graid/datasets/spatank/results/diffusion \
    --gin_config_files /home/spatank/compositional-rl-synth-data/config/diffusion.gin \
    --dataset_type expert \
    --run 20 \
    --num_samples 10000000 \
    --robot IIWA \
    --obj Box \
    --obst None \
    --subtask Trashcan \
    --seed 42
