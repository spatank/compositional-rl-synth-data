"""
Denoiser networks for diffusion.
Code adapts heavily from https://github.com/conglu1997/SynthER.
"""

import math
from typing import Optional

import gin
import torch
import torch.nn as nn
import torch.optim
from einops import rearrange
from torch.nn import functional as F
import numpy as np
from typing import Optional, Sequence, Dict, Any
from abc import ABCMeta, abstractmethod


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
            self,
            dim: int,
            is_random: bool = False,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))


class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "relu",
            layer_norm: bool = False,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
            nn.LayerNorm(width) if layer_norm else torch.nn.Identity(),
        )

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.activation(self.network(x)))


@gin.configurable
class ResidualMLPDenoiser(nn.Module):
    def __init__(
            self,
            d_in: int,
            dim_t: int = 128,
            mlp_width: int = 1024,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = True,
            learned_sinusoidal_dim: int = 16,
            activation: str = "relu",
            layer_norm: bool = True,
            cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.residual_mlp = ResidualMLP(
            input_dim=dim_t,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )
        if cond_dim is not None:
            self.proj = nn.Linear(d_in + cond_dim, dim_t)
            self.conditional = True
        else:
            self.proj = nn.Linear(d_in, dim_t)
            self.conditional = False

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            cond=None,
    ) -> torch.Tensor:
        if self.conditional:
            assert cond is not None
            x = torch.cat((x, cond), dim=-1)
        time_embed = self.time_mlp(timesteps)
        x = self.proj(x) + time_embed
        return self.residual_mlp(x)
    

def adapt_observation_positions(reference_observation_positions: dict, projection_factor: int) -> dict:
    adjusted_observation_positions = {}
    current_position = 0  # this will keep track of the current position across components

    for key, positions in reference_observation_positions.items():
        # For non-ID components, adjust the range of positions based on projection_factor
        if key not in ['object_id', 'robot_id', 'obstacle_id', 'subtask_id']:
            num_components = len(positions)
            new_num_components = num_components * projection_factor
            new_positions = np.arange(current_position, current_position + new_num_components)

            # Update the dictionary with the adjusted positions
            adjusted_observation_positions[key] = new_positions
            current_position += new_num_components  # Update the current_position for the next component
        else:
            # For ID components, make their positions continuous from where the last one ended
            num_components = len(positions)
            new_positions = np.arange(current_position, current_position + num_components)

            adjusted_observation_positions[key] = new_positions
            current_position += num_components  # Update current_position for the next component

    return adjusted_observation_positions


def fanin_init(tensor):
    """Initialize the weights of a layer with fan-in initialization.

    Args:
        tensor (torch.Tensor): Tensor to initialize.

    Returns:
        torch.Tensor: Initialized tensor.

    Raises:
        Exception: If the shape of the tensor is less than 2.
    """
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class _VectorEncoder(nn.Module):

    _observation_shape: Sequence[int]
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool
    _activation: nn.Module
    _feature_size: int
    _fcs: nn.ModuleList
    _bns: nn.ModuleList
    _dropouts: nn.ModuleList

    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self._observation_shape = observation_shape

        if hidden_units is None:
            hidden_units = [256, 256]

        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._feature_size = hidden_units[-1]
        self._activation = activation
        self._use_dense = use_dense

        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        self._fcs = nn.ModuleList()
        self._bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
            if use_dense and i > 0:
                in_unit += observation_shape[0]
            self._fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout(dropout_rate))

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, fc in enumerate(self._fcs):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def last_layer(self) -> nn.Linear:
        return self._fcs[-1]


class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError
    

class CompositionalMLP(nn.Module):
    """Compositional MLP module."""
    def __init__(
        self,
        sizes: Sequence[Sequence[int]],
        num_modules: Sequence[int],
        module_assignment_positions: Sequence[int],
        module_inputs: Sequence[str],
        interface_depths: Sequence[int],
        graph_structure: Sequence[Sequence[int]],
        init_w: float = 3e-3,
        hidden_activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = nn.Identity,
        hidden_init: Optional[nn.Module] = fanin_init,
        b_init_value: float = 0.1,
        layer_norm: bool = False,
        layer_norm_kwargs: Optional[dict] = None,
    ):
        """Initialize the compositional MLP module.
        Args:
            sizes (list): List of sizes of each layer.
            num_modules (list): List of number of modules of each type.
            module_assignment_positions (list): List of module assignment positions.
            module_inputs (list): List of module inputs.
            interface_depths (list): List of interface depths.
            graph_structure (list): List of graph structures.
            init_w (float, optional): Initial weight value. Defaults to 3e-3.
            hidden_activation (nn.Module, optional): Hidden activation module. Defaults to nn.ReLU.
            output_activation (nn.Module, optional): Output activation module. Defaults to nn.Identity.
            hidden_init (function, optional): Hidden initialization function. Defaults to fanin_init.
            b_init_value (float, optional): Initial bias value. Defaults to 0.1.
            layer_norm (bool, optional): Use layer normalization. Defaults to False.
            layer_norm_kwargs (dict, optional): Keyword arguments for layer normalization. Defaults to None.
        """
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.sizes = sizes
        self.num_modules = num_modules
        self.module_assignment_positions = module_assignment_positions
        self.module_inputs = module_inputs  # keys in a dict
        self.interface_depths = interface_depths
        self.graph_structure = (
            graph_structure  # [[0], [1,2], 3] or [[0], [1], [2], [3]]
        )
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.count = 0

        self.module_list = nn.ModuleList()  # e.g., object, robot, task...

        for graph_depth in range(
            len(graph_structure)
        ):  # root -> children -> ... leaves
            for j in graph_structure[
                graph_depth
            ]:  # loop over all module types at this depth
                self.module_list.append(nn.ModuleDict())  # pre, post
                self.module_list[j]["pre_interface"] = nn.ModuleList()
                self.module_list[j]["post_interface"] = nn.ModuleList()
                for k in range(num_modules[j]):  # loop over all modules of this type
                    layers_pre = []
                    layers_post = []
                    for i in range(
                        len(sizes[j]) - 1
                    ):  # loop over all depths in this module
                        if i == interface_depths[j]:
                            input_size = sum(
                                sizes[j_prev][-1]
                                for j_prev in graph_structure[graph_depth - 1]
                            )
                            input_size += sizes[j][i]
                        else:
                            input_size = sizes[j][i]

                        fc = nn.Linear(input_size, sizes[j][i + 1])
                        if (
                            graph_depth < len(graph_structure) - 1
                            or i < len(sizes[j]) - 2
                        ):
                            hidden_init(fc.weight)
                            fc.bias.data.fill_(b_init_value)
                            act = hidden_activation
                            layer_norm_this = layer_norm
                        else:
                            fc.weight.data.uniform_(-init_w, init_w)
                            fc.bias.data.uniform_(-init_w, init_w)
                            act = output_activation
                            layer_norm_this = None

                        if layer_norm_this is not None:
                            new_layer = [fc, nn.LayerNorm(sizes[j][i + 1]), act()]
                        else:
                            new_layer = [fc, act()]

                        if i < interface_depths[j]:
                            layers_pre += new_layer
                        else:
                            layers_post += new_layer
                    if layers_pre:
                        self.module_list[j]["pre_interface"].append(
                            nn.Sequential(*layers_pre)
                        )
                    else:  # it's either a root or a module with no preprocessing
                        self.module_list[j]["pre_interface"].append(nn.Identity())
                    self.module_list[j]["post_interface"].append(
                        nn.Sequential(*layers_post)
                    )

    def forward(self, input_val: torch.Tensor, return_preactivations: bool = False):
        """Forward pass.
        Args:
            input_val (torch.Tensor): Input tensor.
            return_preactivations (bool, optional): Whether to return preactivations. Defaults to False.

        Returns:
            torch.Tensor: Output tensor.
        """
        device = input_val.device
        if len(input_val.shape) > 2:
            input_val = input_val.squeeze(0)
        if return_preactivations:
            raise NotImplementedError("To-do: implement return pre-activations.")
        x = None
        for graph_depth in range(
            len(self.graph_structure)
        ):  # root -> children -> ... -> leaves
            x_post = []  # in case multiple module types at the same depth in the graph
            for j in self.graph_structure[graph_depth]:  # nodes (modules) at this depth
                if len(input_val.shape) == 1:
                    x_pre = input_val[self.module_inputs[j]]
                    onehot = input_val[self.module_assignment_positions[j]]
                    module_index = onehot.nonzero()[0]
                    x_pre = self.module_list[j]["pre_interface"][module_index](x_pre)
                    if x is not None:
                        x_pre = torch.cat((x, x_pre), dim=-1)
                    x_post.append(
                        self.module_list[j]["post_interface"][module_index](x_pre)
                    )
                else:
                    x_post_tmp = torch.empty(input_val.shape[0], self.sizes[j][-1]).to(device)
                    x_pre = input_val[:, self.module_inputs[j]]
                    onehot = input_val[:, self.module_assignment_positions[j]]
                    module_indices = onehot.nonzero(as_tuple=True)
                    assert (
                        module_indices[0]
                        == torch.arange(module_indices[0].shape[0]).to(device)
                        ).all()
                    module_indices_1 = module_indices[1]
                    for module_idx in range(self.num_modules[j]):
                        mask_inputs_for_this_module = module_indices_1 == module_idx
                        mask_to_input_idx = mask_inputs_for_this_module.nonzero()
                        x_pre_this_module = self.module_list[j]["pre_interface"][
                            module_idx
                        ](x_pre[mask_inputs_for_this_module])
                        if x is not None:
                            x_pre_this_module = torch.cat(
                                (x[mask_inputs_for_this_module], x_pre_this_module),
                                dim=-1,
                            )
                        x_post_this_module = self.module_list[j]["post_interface"][
                            module_idx
                        ](x_pre_this_module)
                        mask_to_input_idx = mask_to_input_idx.expand(
                            mask_to_input_idx.shape[0], x_post_this_module.shape[1]
                        )
                        x_post_tmp.scatter_(0, mask_to_input_idx, x_post_this_module)
                    x_post.append(x_post_tmp)
            x = torch.cat(x_post, dim=-1)
        return x


class _CompositionalEncoder(_VectorEncoder):
    """_CompositionalEncoder class."""

    def __init__(
        self,
        encoder_kwargs: dict,
        observation_shape: Sequence[int],
        init_w: float = 3e-3,
    ):
        """Initialize _CompositionalEncoder class.
        Args:
            encoder_kwargs (dict): Encoder parameters.
            observation_shape (Sequence[int]): Observation shape.
            init_w (float, optional): Initial weight. Defaults to 3e-3.
        """
        super().__init__(
            observation_shape,
            hidden_units=None,
            use_batch_norm=False,
            dropout_rate=None,
            use_dense=False,
            activation=nn.ReLU(),
        )

        self._observation_shape = observation_shape
        self.encoder_kwargs = encoder_kwargs
        sizes = encoder_kwargs["sizes"]
        output_dim = encoder_kwargs["output_dim"]
        num_modules = encoder_kwargs["num_modules"]
        module_assignment_positions = encoder_kwargs["module_assignment_positions"]
        module_inputs = encoder_kwargs["module_inputs"]
        interface_depths = encoder_kwargs["interface_depths"]
        graph_structure = encoder_kwargs["graph_structure"]
        sizes = list(sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [output_dim]

        self._feature_size = sizes[-1][-1]

        self.comp_mlp = CompositionalMLP(
            sizes=sizes,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure,
            init_w=init_w,
        )

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.comp_mlp.forward(x)

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError("CompositionalEncoder does not have last_layer")


class CompositionalEncoder(_CompositionalEncoder, Encoder):
    """Implements the actual Compositional Encoder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simply runs the forward pass from _CompositionalEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self._fc_encode(x)


# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class CompositionalResidualBlock(nn.Module):
    def __init__(self, projection_factor, encoder_kwargs, observation_shape, layer_norm):
        super().__init__()

        self.projection_factor = projection_factor

        self.compositional_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=observation_shape,
        )

        self.ln = nn.LayerNorm(164 * self.projection_factor) if layer_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.mlp_linear = nn.Linear(164 * self.projection_factor, 164 * self.projection_factor)


    def forward(self, x):

        # Extract batch components
        current_state = x[:, :(77 * self.projection_factor)]  # Shape [batch_size, 77 * projection_factor]
        action = x[:, (77 * self.projection_factor):(85 * self.projection_factor)]  # Shape [batch_size, 8 * projection_factor]
        reward = x[:, (85 * self.projection_factor):(86 * self.projection_factor)]  # Shape [batch_size, 1 *projection_factor]
        next_state = x[:, (86 * self.projection_factor):(163 * self.projection_factor)]  # Shape [batch_size, 77 * projection_factor]
        terminal_flag = x[:, (163 * self.projection_factor):(164 * self.projection_factor)]  # Shape [batch_size, 1*projection_factor]
        onehot = x[:, (164 * self.projection_factor):(164 * self.projection_factor + 16)]  # Shape [batch_size, 16]

        # Concatenate one-hot vector with current_state and next_state
        current_state_with_onehot = torch.cat([current_state, onehot], dim=-1)  # Shape [batch_size, 93]
        next_state_with_onehot = torch.cat([next_state, onehot], dim=-1)  # Shape [batch_size, 93]

        # Pass through compositional encoder
        current_state_embedding = self.compositional_encoder(current_state_with_onehot)
        next_state_embedding = self.compositional_encoder(next_state_with_onehot)


        learned_residual = torch.cat(
            [current_state_embedding, action, reward, next_state_embedding, terminal_flag], dim=-1
        )
        learned_residual = self.mlp_linear(self.activation(self.ln(learned_residual)))        
        first_164_add = x[:, :164 * self.projection_factor] + learned_residual  # x + learned_residual for the first 164 components

        return torch.cat([first_164_add, onehot], dim=-1)


class CompositionalResidualMLP(nn.Module):
    def __init__(self, 
                 projection_factor,
                 encoder_kwargs, 
                 observation_shape, depth: int, 
                 activation: str = "relu", 
                 layer_norm: bool = False):
        super().__init__()

        self.projection_factor = projection_factor

        self.network = nn.Sequential(
            *[CompositionalResidualBlock(self.projection_factor, encoder_kwargs, observation_shape, layer_norm) for _ in range(depth)]
        )

        # Optional LayerNorm and activation before final output
        self.layer_norm = nn.LayerNorm(164 * self.projection_factor) if layer_norm else nn.Identity()
        self.activation = getattr(F, activation)

        self.final_linear = nn.Linear(164*self.projection_factor, 164)        

        # Independent projection layers for the different components
        self.proj_object = nn.Linear(14, 14 * self.projection_factor)
        self.proj_obstacle = nn.Linear(14, 14 * self.projection_factor)
        self.proj_goal = nn.Linear(17, 17 * self.projection_factor)
        self.proj_proprio = nn.Linear(32, 32 * self.projection_factor)
        self.proj_action = nn.Linear(8, 8 * self.projection_factor)
        self.proj_reward = nn.Linear(1, 1 * self.projection_factor)
        self.proj_terminal_flag = nn.Linear(1, 1 * self.projection_factor)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        current_state = x[:, :77]  # Shape [batch_size, 77]
        action = x[:, 77:85]  # Shape [batch_size, 8]
        reward = x[:, 85:86]  # Shape [batch_size]
        next_state = x[:, 86:163]  # Shape [batch_size, 77]
        terminal_flag = x[:, 163:164]  # Shape [batch_size]
        onehot = x[:, 164:180]  # Shape [batch_size, 16]

        current_state_object = current_state[:, :14]
        current_state_obstacle = current_state[:, 14:28]
        current_state_goal = current_state[:, 28:45]
        current_state_proprio = current_state[:, 45:77]

        next_state_object = next_state[:, :14]
        next_state_obstacle = next_state[:, 14:28]
        next_state_goal = next_state[:, 28:45]
        next_state_proprio = next_state[:, 45:77]

        projected_current_state_object = self.proj_object(current_state_object)
        projected_current_state_obstacle = self.proj_obstacle(current_state_obstacle)
        projected_current_state_goal = self.proj_goal(current_state_goal)
        projected_current_state_proprio = self.proj_proprio(current_state_proprio)

        projected_next_state_object = self.proj_object(next_state_object)
        projected_next_state_obstacle = self.proj_obstacle(next_state_obstacle)
        projected_next_state_goal = self.proj_goal(next_state_goal)
        projected_next_state_proprio = self.proj_proprio(next_state_proprio)

        projected_current_state = torch.cat([projected_current_state_object, projected_current_state_obstacle, projected_current_state_goal, projected_current_state_proprio], dim = -1)
        projected_next_state = torch.cat([projected_next_state_object, projected_next_state_obstacle, projected_next_state_goal, projected_next_state_proprio], dim = -1)

        projected_action = self.proj_action(action)
        projected_reward = self.proj_reward(reward)
        projected_terminal_flag = self.proj_terminal_flag(terminal_flag)

        projected_input = torch.cat([projected_current_state, projected_action, projected_reward, projected_next_state, projected_terminal_flag, onehot], dim=-1)

        repeated_residual_output = self.network(projected_input)

        output = self.final_linear(self.activation(self.layer_norm(repeated_residual_output[:, :self.projection_factor * 164])))  # apply LayerNorm and non-linearity

        return output


@gin.configurable
class CompositionalResidualMLPDenoiser(nn.Module):
    def __init__(
            self,
            projection_factor,
            d_in: int,
            dim_t: int = 164,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = True,
            learned_sinusoidal_dim: int = 16,
            cond_dim: Optional[int] = None,
    ):
        super().__init__()

        reference_observation_positions = {
            'object-state': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 
            'obstacle-state': np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 
            'goal-state': np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]),
            'robot0_proprio-state': np.array([45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                            62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]),
            'object_id': np.array([77, 78, 79, 80]), 
            'robot_id': np.array([81, 82, 83, 84]), 
            'obstacle_id': np.array([85, 86, 87, 88]), 
            'subtask_id': np.array([89, 90, 91, 92])
        }

        self.projection_factor = projection_factor  # set desired projection factor
        observation_positions = adapt_observation_positions(reference_observation_positions, self.projection_factor)

        sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))
        module_names = ["obstacle_id", "object_id", "subtask_id", "robot_id"]
        module_input_names = ["obstacle-state", "object-state", "goal-state", "robot0_proprio-state"]

        module_assignment_positions = [observation_positions[key] for key in module_names]
        interface_depths = [-1, 1, 2, 3]
        graph_structure = [[0], [1], [2], [3]]
        num_modules = [len(onehot_pos) for onehot_pos in module_assignment_positions]

        module_inputs = []
        for key in module_input_names:
            if isinstance(key, list):
                # concatenate the inputs
                module_inputs.append(
                    np.concatenate([observation_positions[k] for k in key], axis=0)
                )
            else:
                module_inputs.append(observation_positions[key])

        obs_dim = 77 * self.projection_factor + 16
        act_dim = 8 * self.projection_factor
        output_dim = 77 * self.projection_factor
        observation_shape = (obs_dim,)

        encoder_kwargs = {
                "sizes": sizes,
                "obs_dim": obs_dim,
                "output_dim": output_dim if output_dim is not None else act_dim,
                "num_modules": num_modules,
                "module_assignment_positions": module_assignment_positions,
                "module_inputs": module_inputs,
                "interface_depths": interface_depths,
                "graph_structure": graph_structure,
            }

        self.residual_mlp = CompositionalResidualMLP(
            projection_factor=self.projection_factor,
            encoder_kwargs=encoder_kwargs,
            observation_shape=observation_shape,
            depth=num_layers
        )
        if cond_dim is not None:
            self.proj = nn.Linear(d_in + cond_dim, dim_t)
            self.conditional = True
        else:
            self.proj = nn.Linear(d_in, dim_t)
            self.conditional = False

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            cond=None,
    ) -> torch.Tensor:

        time_embed = self.time_mlp(timesteps)
        x = x + time_embed
        if self.conditional:
            assert cond is not None
            x = torch.cat((x, cond), dim=-1)

        return self.residual_mlp(x)
