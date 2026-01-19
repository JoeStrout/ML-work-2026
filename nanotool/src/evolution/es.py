"""
Evolution Strategies implementation.

This module provides OpenAI-ES and related evolutionary optimization
algorithms for training neural networks without gradients.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import torch


class EvolutionStrategy:
    """
    OpenAI-style Evolution Strategy optimizer.

    Based on Salimans et al. (2017) "Evolution Strategies as a
    Scalable Alternative to Reinforcement Learning"
    """

    def __init__(
        self,
        num_params: int,
        sigma: float = 0.5,
        learning_rate: float = 0.5,
        population_size: int = 50,
        antithetic: bool = False,
        weight_decay: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize ES optimizer.

        Args:
            num_params: Number of parameters to optimize
            sigma: Standard deviation of perturbations
            learning_rate: Step size for parameter updates
            population_size: Number of perturbations per generation
            antithetic: Use mirrored sampling (recommended)
            weight_decay: L2 regularization coefficient
            seed: Random seed for reproducibility
        """
        self.num_params = num_params
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.population_size = population_size
        self.antithetic = antithetic
        self.weight_decay = weight_decay

        if seed is not None:
            np.random.seed(seed)

        # Current parameters (will be set externally or initialized)
        self.theta = np.zeros(num_params)

    def ask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate population of perturbed parameters for evaluation.

        Returns:
            perturbations: Array of noise vectors (population_size, num_params)
            population: Array of perturbed parameters (population_size, num_params)
        """
        if self.antithetic:
            # Generate half the perturbations, mirror for other half
            half_pop = self.population_size // 2
            epsilon_half = np.random.randn(half_pop, self.num_params)
            epsilon = np.concatenate([epsilon_half, -epsilon_half], axis=0)
        else:
            epsilon = np.random.randn(self.population_size, self.num_params)

        population = self.theta + self.sigma * epsilon
        return epsilon, population

    def tell(self, epsilon: np.ndarray, fitness: np.ndarray) -> float:
        """
        Update parameters based on fitness evaluations.

        Args:
            epsilon: Perturbations used (from ask())
            fitness: Fitness values for each perturbation (higher is better)

        Returns:
            Mean fitness of this generation
        """
        # Rank-based fitness shaping (more robust than z-score)
        # Higher rank = better fitness
        ranks = np.zeros(len(fitness))
        sorted_indices = np.argsort(fitness)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank

        # Normalize ranks to [-0.5, 0.5] centered at 0
        ranks = (ranks / (len(fitness) - 1)) - 0.5

        # Compute gradient estimate using ranks instead of raw fitness
        # Note: must normalize by population size
        gradient = np.dot(epsilon.T, ranks) / (self.population_size * self.sigma)

        # Apply weight decay
        if self.weight_decay > 0:
            self.theta *= (1 - self.weight_decay)

        # Update parameters
        self.theta += self.learning_rate * gradient

        return np.mean(fitness)

    def get_params(self) -> np.ndarray:
        """Get current parameter vector."""
        return self.theta.copy()

    def set_params(self, theta: np.ndarray):
        """Set parameter vector."""
        self.theta = theta.copy()


def flatten_params(model: torch.nn.Module) -> np.ndarray:
    """Extract all parameters from a PyTorch model as a flat numpy array."""
    params = []
    for p in model.parameters():
        params.append(p.data.cpu().numpy().flatten())
    return np.concatenate(params)


def unflatten_params(model: torch.nn.Module, flat_params: np.ndarray):
    """Load flat numpy array back into PyTorch model parameters."""
    idx = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = torch.from_numpy(
            flat_params[idx:idx + num_params].reshape(p.shape)
        ).float().to(p.device)
        idx += num_params
