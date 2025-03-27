#!/bin/bash
#SBATCH --job-name=policy_training
#SBATCH --output=slurm/%j.out
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/spatank/compositional-rl-synth-data/venv/bin/activate

python /home/spatank/compositional-rl-synth-data/scripts/train_policy.py \
    --base_agent_data_path /home/spatank/compositional-rl-synth-data/data \
    --base_synthetic_data_path /home/spatank/compositional-rl-synth-data/results/diffusion \
    --base_results_folder /home/spatank/compositional-rl-synth-data/results/policies \
    --dataset_type synthetic \
    --robot IIWA \
    --obj Box \
    --obst ObjectDoor \
    --subtask Push \
    --algorithm iql \
    --seed 0 \
    --denoiser monolithic \
    --task_list_seed 0 \
    --num_train 56 \
    --diffusion_training_run 1
