import argparse
import pathlib
import torch
import numpy as np
import gin
from diffusion.utils import *
from diffusion.train_diffuser import SimpleDiffusionGenerator
import composuite

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_data_path', type=str, required=True, help='Base path to datasets.')
    parser.add_argument('--base_results_folder', type=str, required=True, help='Base path to results.')

    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/diffusion.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[], help='Additional gin parameters.')

    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., expert data).')
    parser.add_argument('--run', type=int, default=0, help='Diffusion training run ID.')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of transitions to generate.')

    parser.add_argument('--robot', type=str, required=True, help='Robot name.')
    parser.add_argument('--obj', type=str, required=True, help='Object name.')
    parser.add_argument('--obst', type=str, required=True, help='Obstacle name.')
    parser.add_argument('--subtask', type=str, required=True, help='Subtask name')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    base_results_path = pathlib.Path(args.base_results_folder)
    results_folder = base_results_path / f"cond_diff_{args.run}"
    assert results_folder.exists(), f"The results folder associated with run {args.run} does not exist."

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    checkpoint_path = os.path.join(results_folder, 'model-100000.pt')

    robot, obj, obst, subtask = args.robot, args.obj, args.obst, args.subtask
    subtask_folder = os.path.join(results_folder, f"{robot}_{obj}_{obst}_{subtask}")

    representative_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=False, ignore_done=False)
    representative_indicators_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=True, ignore_done=False)
    modality_dims = representative_indicators_env.modality_dims

    dataset = load_single_composuite_dataset(args.base_data_path, args.dataset_type, robot, obj, obst, subtask)
    dataset = transitions_dataset(dataset)
    dataset, indicators = remove_indicator_vectors(modality_dims, dataset)
    inputs = make_inputs(dataset)

    inputs = torch.from_numpy(inputs).float()
    indicators = torch.from_numpy(indicators).float()
    dataset = torch.utils.data.TensorDataset(inputs, indicators)

    diffusion = construct_diffusion_model(inputs=inputs, cond_dim=indicators.shape[1])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    ema_dict = checkpoint['ema']
    ema_dict = {k: v for k, v in ema_dict.items() if k.startswith('ema_model')}
    ema_dict = {k.replace('ema_model.', ''): v for k, v in ema_dict.items()}
    diffusion.load_state_dict(ema_dict)
    diffusion.eval()

    subtask_indicator = get_task_indicator(robot, obj, obst, subtask)
    generator = SimpleDiffusionGenerator(env=representative_env, ema_model=diffusion)
    obs, actions, rewards, next_obs, terminals = generator.sample(num_samples=args.num_samples, cond=subtask_indicator)

    np.savez_compressed(
        pathlib.Path(subtask_folder) / 'large_samples_dataset.npz',
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminals=terminals
    )
