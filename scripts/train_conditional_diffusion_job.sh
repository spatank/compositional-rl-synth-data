#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=slurm/%A_%a_train_conditional_diffusion.out
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/spatank/compositional-rl-synth-data/venv/bin/activate

python /home/spatank/compositional-rl-synth-data/scripts/train_conditional_diffusion.py \
    --base_data_path /mnt/kostas-graid/datasets/spatank \
    --base_results_folder /home/spatank/compositional-rl-synth-data/results \
    --gin_config_files /home/spatank/compositional-rl-synth-data/config/diffusion.gin \
    --dataset_type expert \
    --robots Panda \
    --objs Box Dumbbell Plate Hollowbox \
    --obsts None GoalWall ObjectDoor ObjectWall \
    --tasks PickPlace Push Shelf Trashcan \
    --seed 42
