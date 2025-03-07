#!/bin/bash

remote_host="spatank@grasp-login2.seas.upenn.edu"
base_remote_folder="/home/spatank/compositional-rl-synth-data/results/diffusion/non_comp_diff_2"
local_destination="/Users/shubhankar/Developer/compositional-rl-synth-data/cluster_results/diffusion/non_comp_diff_2/train"

mkdir -p "$local_destination"

while IFS= read -r folder || [[ -n "$folder" ]]; do
    remote_folder="$base_remote_folder/$folder"
    folder_name=$(basename "$remote_folder")
    mkdir -p "$local_destination/$folder_name"
    rsync -avz -e ssh "$remote_host:$remote_folder/" "$local_destination/$folder_name/"
done < train_task_folders_to_move.txt

echo "Transfer complete!"
