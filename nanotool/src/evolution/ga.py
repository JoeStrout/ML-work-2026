"""
Genetic Algorithm with batched GPU evaluation.

This module provides a simple GA that evaluates the entire population
in parallel on GPU using batched matrix operations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class GAConfig:
    """Configuration for the Genetic Algorithm."""
    population_size: int = 100
    elite_fraction: float = 0.1  # Top 10% survive unchanged
    tournament_size: int = 3
    mutation_rate: float = 1.0  # Probability of mutating each individual
    mutation_std: float = 0.1   # Standard deviation of Gaussian mutation
    crossover_rate: float = 0.0  # Probability of crossover (often harmful for NN weights)
    seed: Optional[int] = None


class BatchedMLP(torch.nn.Module):
    """
    MLP that can evaluate multiple weight sets simultaneously.

    Instead of a single set of weights, maintains a population of weight sets
    and computes forward passes for all of them in parallel using batched matmul.
    """

    def __init__(
        self,
        layer_sizes: list,
        population_size: int,
        activation: str = 'silu',
        device: str = 'cuda'
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.population_size = population_size
        self.device = device
        self.num_layers = len(layer_sizes) - 1

        # Activation function
        self.activation_name = activation
        if activation == 'silu':
            self.activation = F.silu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = lambda x: F.leaky_relu(x, 0.1)
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x  # linear

        # Initialize population of weights
        # weights[i] has shape (population_size, out_features, in_features)
        # biases[i] has shape (population_size, out_features)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]

            # Xavier/Glorot initialization
            std = np.sqrt(2.0 / (in_features + out_features))
            w = torch.randn(population_size, out_features, in_features, device=device) * std
            b = torch.zeros(population_size, out_features, device=device)

            self.weights.append(w)
            self.biases.append(b)

        self._num_params = sum(
            w.shape[1] * w.shape[2] + b.shape[1]
            for w, b in zip(self.weights, self.biases)
        )

    @property
    def num_params_per_member(self) -> int:
        """Number of parameters per population member."""
        return self._num_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all population members simultaneously.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (population_size, batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Expand x to (population_size, batch_size, input_dim)
        h = x.unsqueeze(0).expand(self.population_size, -1, -1)

        for i in range(self.num_layers):
            # h: (pop, batch, in_features)
            # w: (pop, out_features, in_features)
            # We want: (pop, batch, out_features)

            # Use einsum for batched matmul: (pop, batch, in) @ (pop, in, out) -> (pop, batch, out)
            # Note: weights are (pop, out, in), so we transpose
            h = torch.einsum('pbi,poi->pbo', h, self.weights[i])

            # Add bias: (pop, out) -> broadcast to (pop, batch, out)
            h = h + self.biases[i].unsqueeze(1)

            # Apply activation (except last layer)
            if i < self.num_layers - 1:
                h = self.activation(h)

        return h

    def get_member_params(self, idx: int) -> np.ndarray:
        """Get flattened parameters for a single population member."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w[idx].cpu().numpy().flatten())
            params.append(b[idx].cpu().numpy().flatten())
        return np.concatenate(params)

    def set_member_params(self, idx: int, params: np.ndarray):
        """Set parameters for a single population member from flattened array."""
        offset = 0
        for i in range(self.num_layers):
            w_shape = self.weights[i][idx].shape
            w_size = w_shape[0] * w_shape[1]
            self.weights[i][idx] = torch.from_numpy(
                params[offset:offset + w_size].reshape(w_shape)
            ).float().to(self.device)
            offset += w_size

            b_shape = self.biases[i][idx].shape
            b_size = b_shape[0]
            self.biases[i][idx] = torch.from_numpy(
                params[offset:offset + b_size]
            ).float().to(self.device)
            offset += b_size

    def get_all_params(self) -> torch.Tensor:
        """Get all parameters as a single tensor (population_size, num_params)."""
        params_list = []
        for w, b in zip(self.weights, self.biases):
            # w: (pop, out, in) -> (pop, out*in)
            params_list.append(w.view(self.population_size, -1))
            # b: (pop, out)
            params_list.append(b)
        return torch.cat(params_list, dim=1)

    def set_all_params(self, params: torch.Tensor):
        """Set all parameters from a tensor (population_size, num_params)."""
        offset = 0
        for i in range(self.num_layers):
            w_shape = self.weights[i].shape
            w_size = w_shape[1] * w_shape[2]
            self.weights[i] = params[:, offset:offset + w_size].view(w_shape)
            offset += w_size

            b_size = self.biases[i].shape[1]
            self.biases[i] = params[:, offset:offset + b_size]
            offset += b_size


class GeneticAlgorithm:
    """
    Simple Genetic Algorithm with tournament selection and Gaussian mutation.

    Designed to work with BatchedMLP for efficient GPU evaluation.
    """

    def __init__(self, config: GAConfig, model: BatchedMLP):
        self.config = config
        self.model = model
        self.population_size = config.population_size
        self.device = model.device

        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        # Track best individual
        self.best_fitness = float('-inf')
        self.best_params = None

        # Statistics
        self.generation = 0

    def evaluate(self, x: torch.Tensor, fitness_fn: Callable) -> torch.Tensor:
        """
        Evaluate fitness of all population members.

        Args:
            x: Input batch (batch_size, input_dim)
            fitness_fn: Function(outputs, x) -> fitness tensor (population_size,)
                       Higher fitness is better.

        Returns:
            Fitness tensor of shape (population_size,)
        """
        with torch.no_grad():
            outputs = self.model(x)  # (pop, batch, output_dim)
            fitness = fitness_fn(outputs, x)
        return fitness

    def step(self, fitness: torch.Tensor) -> float:
        """
        Perform one generation of evolution.

        Args:
            fitness: Fitness values (population_size,), higher is better

        Returns:
            Mean fitness of the population
        """
        fitness_np = fitness.cpu().numpy()

        # Track best
        best_idx = np.argmax(fitness_np)
        if fitness_np[best_idx] > self.best_fitness:
            self.best_fitness = fitness_np[best_idx]
            self.best_params = self.model.get_member_params(best_idx).copy()

        # Get current parameters
        params = self.model.get_all_params()  # (pop, num_params)

        # Selection and reproduction
        new_params = self._reproduce(params, fitness_np)

        # Update model
        self.model.set_all_params(new_params)

        self.generation += 1
        return float(np.mean(fitness_np))

    def _reproduce(self, params: torch.Tensor, fitness: np.ndarray) -> torch.Tensor:
        """Create next generation through selection, crossover, and mutation."""
        pop_size = self.population_size
        num_params = params.shape[1]

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness)[::-1]

        # Elite selection: top individuals survive unchanged
        num_elite = max(1, int(pop_size * self.config.elite_fraction))
        elite_indices = sorted_indices[:num_elite]

        new_params = torch.zeros_like(params)

        # Copy elites
        for i, elite_idx in enumerate(elite_indices):
            new_params[i] = params[elite_idx]

        # Fill rest through tournament selection + mutation
        for i in range(num_elite, pop_size):
            # Tournament selection
            parent_idx = self._tournament_select(fitness)
            child = params[parent_idx].clone()

            # Optional crossover with another parent
            if np.random.random() < self.config.crossover_rate:
                parent2_idx = self._tournament_select(fitness)
                child = self._crossover(child, params[parent2_idx])

            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child = self._mutate(child)

            new_params[i] = child

        return new_params

    def _tournament_select(self, fitness: np.ndarray) -> int:
        """Select individual via tournament selection."""
        tournament_indices = np.random.choice(
            self.population_size,
            size=self.config.tournament_size,
            replace=False
        )
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx

    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        """Uniform crossover between two parents."""
        mask = torch.rand_like(parent1) < 0.5
        child = torch.where(mask, parent1, parent2)
        return child

    def _mutate(self, params: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian mutation."""
        noise = torch.randn_like(params) * self.config.mutation_std
        return params + noise

    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get best individual found so far."""
        return self.best_params, self.best_fitness


def create_autoencoder_fitness_fn(num_digits: int):
    """
    Create a fitness function for autoencoder task.

    Returns negative MSE loss (higher is better).
    """
    def fitness_fn(outputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # outputs: (pop, batch, output_dim)
        # x: (batch, input_dim)

        # Expand x to match outputs shape
        x_expanded = x.unsqueeze(0).expand_as(outputs)

        # MSE per population member
        mse = ((outputs - x_expanded) ** 2).mean(dim=(1, 2))

        # Return negative MSE (higher is better)
        return -mse

    return fitness_fn


def compute_batch_accuracy(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """
    Compute digit accuracy for each population member.

    Args:
        outputs: (population_size, batch_size, num_digits * 10)
        x: (batch_size, num_digits * 10)
        num_digits: Number of digits

    Returns:
        Accuracy per population member (population_size,)
    """
    pop_size, batch_size, _ = outputs.shape

    # Reshape to (pop, batch, digits, 10)
    outputs_reshaped = outputs.view(pop_size, batch_size, num_digits, 10)
    x_reshaped = x.view(batch_size, num_digits, 10)

    # Get predicted and true digits
    pred_digits = outputs_reshaped.argmax(dim=3)  # (pop, batch, digits)
    true_digits = x_reshaped.argmax(dim=2)  # (batch, digits)

    # Expand true_digits for comparison
    true_digits_expanded = true_digits.unsqueeze(0).expand_as(pred_digits)

    # Digit accuracy per population member
    correct = (pred_digits == true_digits_expanded).float()
    accuracy = correct.mean(dim=(1, 2))  # Average over batch and digits

    return accuracy
