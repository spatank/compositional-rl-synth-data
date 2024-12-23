import argparse
import pathlib
from collections import defaultdict
import torch
import wandb
import numpy as np
import gin
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

    # Environment
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., expert data).')
    parser.add_argument('--robots', nargs='+', type=str, required=True, help='List of robots.')
    parser.add_argument('--objs', nargs='+', type=str, required=True, help='List of objects.')
    parser.add_argument('--obsts', nargs='+', type=str, required=True, help='List of obstacles.')
    parser.add_argument('--tasks', nargs='+', type=str, required=True, help='List of tasks.')
    parser.add_argument('--exclude', nargs='+', type=str, default=[], help='List of environments to exclude.')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    # W&B config
    parser.add_argument('--wandb_project', type=str, default="offline_rl_diffusion")
    parser.add_argument('--wandb_entity', type=str, default="")
    parser.add_argument('--wandb_group', type=str, default="diffusion_training")
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    base_results_path = pathlib.Path(args.base_results_folder)
    idx = 1
    while (base_results_path / f"cluster_multidata_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"cluster_multidata_{idx}"
    results_folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    exclude_set = set()
    if args.exclude:
        exclude_set = {tuple(item.split('-')) for item in args.exclude}
    combinations = list(product(args.robots, args.objs, args.obsts, args.tasks))
    combinations_subset = [combination for combination in combinations if combination not in exclude_set]
    datasets = []
    for combination in tqdm(combinations_subset, desc="Loading data"):
        print('Loading:', combination)
        robot, obj, obst, task = combination
        datasets.append(load_single_composuite_dataset(args.base_data_path, args.dataset_type, robot, obj, obst, task))
    
    datasets = [transitions_dataset(dataset) for dataset in datasets]

    combined_data_dict = defaultdict(list)

    for idx, data in enumerate(datasets):
        for key in data.keys():
            combined_data_dict[key].append(data[key])

    combined_transitions_datasets = {key: np.concatenate(values, axis=0) for key, values in combined_data_dict.items()}
    print('Building training data.')
    inputs = make_inputs(combined_transitions_datasets)
    print('Input shape before removing task indicators:', inputs.shape)
    env = composuite.make(args.robots, args.objs, args.obsts, args.tasks, use_task_id_obs=True, ignore_done=False)
    inputs, indicators = remove_indicator_vectors(inputs, env)
    print('Input shape after removing task indicators:', inputs.shape)

    inputs = torch.from_numpy(inputs).float()
    indicators = torch.from_numpy(indicators).float()
    dataset = torch.utils.data.TensorDataset(inputs, indicators)

    diffusion = construct_diffusion_model(inputs=inputs, cond_dim=indicators.shape[1])

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=results_folder.name,
    )

    # Trainer
    trainer = Trainer(diffusion, dataset, results_folder=str(results_folder))
    trainer.train()

    for robot, obj, obst, task in combinations:
        print('Generating synthetic data:', robot, obj, obst, task)
        task_folder = results_folder / f"{robot}_{obj}_{obst}_{task}"
        task_folder.mkdir(parents=True, exist_ok=True)

        task_indicator = get_task_indicator(robot, obj, obst, task)
        env = composuite.make(robot, obj, obst, task, use_task_id_obs=False, ignore_done=False)
        generator = SimpleDiffusionGenerator(env=env, ema_model=trainer.ema.ema_model)
        obs, actions, rewards, next_obs, terminals = generator.sample(num_samples=100000, cond=task_indicator)

        np.savez_compressed(
            task_folder / 'samples.npz',
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            terminals=terminals
        )
        