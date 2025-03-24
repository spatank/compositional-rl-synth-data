import argparse
import pathlib
from collections import defaultdict
import torch
import wandb
import numpy as np
import gin
import pickle
from diffusion.utils import *
from diffusion.elucidated_diffusion import Trainer
from diffusion.train_diffuser import SimpleDiffusionGenerator
import composuite

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_data_path', type=str, required=True, help='Base path to datasets.')
    parser.add_argument('--base_results_folder', type=str, required=True, help='Base path to results.')
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/diffusion.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[], help='Additional gin parameters.')
    
    parser.add_argument('--denoiser', type=str, default='monolithic', help='Type of denoiser network.')
    parser.add_argument('--task_list_path', type=str, required=True, help='Path to task splits.')
    parser.add_argument('--num_train', type=int, required=True, help='Number of training tasks.')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., expert data).')
    parser.add_argument('--experiment_type', type=str, required=True, help='CompoSuite experiment type.', default='default')
    parser.add_argument('--element', type=str, required=False, help='CompoSuite element.')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument('--wandb_project', type=str, default="diffusion_training")
    
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    base_results_path = pathlib.Path(args.base_results_folder)
    idx = 1
    folder_prefix = f"{args.denoiser}_seed{args.seed}_train{args.num_train}"
    while (base_results_path / f"{folder_prefix}_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"{folder_prefix}_{idx}"
    results_folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    exp_name, train_task_list, _, _ = get_task_list(
        args.task_list_path,
        args.dataset_type,
        args.experiment_type,
        None,  # holdout element
        args.seed,
    )
    train_task_list = [tuple(task) for task in train_task_list]
    train_task_list = train_task_list[:args.num_train]        

    datasets = []
    for task in tqdm(train_task_list, desc="Loading data"):
        print('Loading:', task)
        robot, obj, obst, subtask = task
        datasets.append(load_single_composuite_dataset(args.base_data_path, args.dataset_type, robot, obj, obst, subtask))
    datasets = [transitions_dataset(dataset) for dataset in datasets]

    combined_data_dict = defaultdict(list)
    for idx, data in enumerate(datasets):
        for key in data.keys():
            combined_data_dict[key].append(data[key])

    combined_transitions_datasets = {key: np.concatenate(values, axis=0) for key, values in combined_data_dict.items()}
    print('Building training data.')

    inputs = make_inputs(combined_transitions_datasets)
    inputs = torch.from_numpy(inputs).float()
    dataset = torch.utils.data.TensorDataset(inputs)

    diffusion = construct_diffusion_model(inputs=inputs, denoiser='monolithic')

    wandb.init(
        project=args.wandb_project,
        group=f"{args.denoiser}_seed{args.seed}",  # group by denoiser type and seed
        name=results_folder.name,
        tags=[args.denoiser, f"seed_{args.seed}", f"train_{args.num_train}", args.dataset_type],
        config={
            "denoiser": args.denoiser,
            "seed": args.seed,
            "num_train": args.num_train,
            "dataset_type": args.dataset_type,
            "experiment_type": args.experiment_type,
        }
    )

    # Trainer
    trainer = Trainer(diffusion, dataset, results_folder=str(results_folder))
    trainer.train()

    for robot, obj, obst, subtask in train_task_list:
        print('Generating synthetic data:', robot, obj, obst, subtask)
        subtask_folder = results_folder / f"{robot}_{obj}_{obst}_{subtask}"
        subtask_folder.mkdir(parents=True, exist_ok=True)

        env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=True, ignore_done=False)
        generator = SimpleDiffusionGenerator(env=env, ema_model=trainer.ema.ema_model)
        observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=100000)

        np.savez_compressed(
            subtask_folder / 'samples.npz',
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals
        )
