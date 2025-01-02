import argparse
import pathlib
from collections import defaultdict
import torch
import wandb
import numpy as np
import gin
import pickle
from diffusion.utils import *
import composuite
from diffusion.utils import *
from CORL.algorithms.offline.td3_bc import *
from CORL.shared.buffer import *
from CORL.shared.logger import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_data_path', type=str, required=True, help='Base path to datasets.')
    parser.add_argument('--base_results_folder', type=str, required=True, help='Base path to results.')

    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/diffusion.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[], help='Additional gin parameters.')

    # Environment
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., expert data).')
    parser.add_argument('--experiment_type', type=str, required=True, help='CompoSuite experiment type.', default='default')
    parser.add_argument('--element', type=str, required=False, help='CompoSuite element.')
    parser.add_argument('--num_train', type=int, required=True, help='Number of CompoSuite tasks.')

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
    while (base_results_path / f"offline_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"offline_{idx}"
    results_folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    if args.experiment_type == 'default':
        train_tasks, _ = composuite.sample_tasks(experiment_type='default', num_train=args.num_train)
    if args.experiment_type == 'smallscale':
        element = args.element
        train_tasks, _ = composuite.sample_tasks(experiment_type='smallscale', 
                                                 smallscale_elem=args.element, num_train=args.num_train)
    if args.experiment_type == 'holdout':
        train_tasks, _ = composuite.sample_tasks(experiment_type='holdout', 
                                                 holdout_elem=args.element, num_train=args.num_train)
        
    with open(results_folder / "tasks.pkl", 'wb') as file:
        pickle.dump(train_tasks, file)

    datasets = []
    for task in tqdm(train_tasks, desc="Loading data"):
        print('Loading:', task)
        robot, obj, obst, subtask = task
        datasets.append(load_single_composuite_dataset(args.base_data_path, args.dataset_type, robot, obj, obst, subtask))
    datasets = [transitions_dataset(dataset) for dataset in datasets]

    combined_data_dict = defaultdict(list)
    for idx, data in enumerate(datasets):
        for key in data.keys():
            combined_data_dict[key].append(data[key])

    combined_transitions_datasets = {key: np.concatenate(values, axis=0) for key, values in combined_data_dict.items()}

    robot, obj, obst, subtask = train_tasks[0]
    env = composuite.make(robot, obj, obst, subtask, 
                          use_task_id_obs=True, ignore_done=False)
    state_mean, state_std = compute_mean_std(combined_transitions_datasets["observations"], eps=1e-3)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    config = TrainConfig()

    config.device = "cuda"
    config.env = "iiwa-box-none-push"
    config.diffusion.path = "/Users/shubhankar/Developer/compositional-rl-synth-data/results/Diffusion/IIWA_Box_None_Push/samples.npz"

    replay_buffer = prepare_replay_buffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        dataset=dataset,
        env_name=config.env,
        device=config.device,
        reward_normalizer=RewardNormalizer(dataset, config.env) if config.normalize_reward else None,
        state_normalizer=StateNormalizer(state_mean, state_std),
        diffusion_config=config.diffusion
        )



    print('Building training data.')
    inputs = make_inputs(combined_transitions_datasets)
    inputs = torch.from_numpy(inputs).float()
    dataset = torch.utils.data.TensorDataset(inputs)

    diffusion = construct_diffusion_model(inputs=inputs)

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

        env = composuite.make(robot, obj, obst, task, use_task_id_obs=True, ignore_done=False)
        generator = SimpleDiffusionGenerator(env=env, ema_model=trainer.ema.ema_model)
        observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=100000)

        np.savez_compressed(
            task_folder / 'samples.npz',
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals
        )