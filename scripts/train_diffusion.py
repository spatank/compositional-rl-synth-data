import argparse
import pathlib
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
    parser.add_argument('--robot', type=str, default='IIWA', help='Robot type for CompoSuite.')
    parser.add_argument('--obj', type=str, default='Hollowbox', help='Object type for task.')
    parser.add_argument('--obst', type=str, default='None', help='Obstacle type for task.')
    parser.add_argument('--task', type=str, default='PickPlace', help='Task type.')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    # W&B config
    parser.add_argument('--wandb_project', type=str, default="offline_rl_diffusion")
    parser.add_argument('--wandb_entity', type=str, default="")
    parser.add_argument('--wandb_group', type=str, default="diffusion_training")
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    results_folder = pathlib.Path(args.base_results_folder) / f"{args.robot}_{args.obj}_{args.obst}_{args.task}"
    results_folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    dataset = load_single_composuite_dataset(
        base_path=args.base_data_path,
        dataset_type=args.dataset_type,
        robot=args.robot,
        obj=args.obj,
        obst=args.obst,
        task=args.task
    )
    dataset = transitions_dataset(dataset)
    inputs = make_inputs(dataset)
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

    env = composuite.make(args.robot, args.obj, args.obst, args.task, use_task_id_obs=True, ignore_done=False)
    generator = SimpleDiffusionGenerator(env=env, ema_model=trainer.ema.ema_model)
    observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=100000)

    np.savez_compressed(
        results_folder / 'samples.npz',
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals
    )