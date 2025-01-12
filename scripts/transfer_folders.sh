#!/bin/bash

remote_host="spatank@grasp-login2.seas.upenn.edu"
base_remote_folder="/mnt/kostas-graid/datasets/spatank/results/diffusion/cond_diff_21"
local_destination="/Users/shubhankar/Developer/compositional-rl-synth-data/cluster_results/diffusion/cond_diff_21/test"

mkdir -p "$local_destination"

# Ensure the file doesn't have an empty last line
while IFS= read -r folder || [[ -n "$folder" ]]; do
    remote_folder="$base_remote_folder/$folder"
    folder_name=$(basename "$remote_folder")
    mkdir -p "$local_destination/$folder_name"
    rsync -avz -e ssh "$remote_host:$remote_folder/" "$local_destination/$folder_name/"
done < test_task_folders_to_move.txt

echo "Transfer complete!"
