"""
Training loop for evolution strategies.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import time
import json
from pathlib import Path

from ..evolution.es import EvolutionStrategy, flatten_params, unflatten_params
from ..dataset.addition import generate_addition_batch
from ..encodings.numbers import DigitEncoding


@dataclass
class TrainingConfig:
    """Configuration for ES training."""
    # Task
    num_digits: int = 4
    batch_size: int = 128

    # ES hyperparameters
    population_size: int = 50
    sigma: float = 0.02
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    generations: int = 500

    # Evaluation
    eval_batch_size: int = 1000
    eval_frequency: int = 10

    # Misc
    seed: int = 42
    device: str = 'cpu'
    save_dir: Optional[str] = None
    checkpoint_frequency: int = 50


class ESTrainer:
    """
    Evolution Strategy trainer for addition networks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Initialize ES
        self.params = flatten_params(self.model)
        self.es = EvolutionStrategy(
            num_params=len(self.params),
            sigma=config.sigma,
            learning_rate=config.learning_rate,
            population_size=config.population_size,
            weight_decay=config.weight_decay,
            seed=config.seed
        )
        self.es.set_params(self.params)

        # Set up encodings for evaluation
        self.output_encoding = DigitEncoding(config.num_digits + 1)

        # History tracking
        self.history = {
            'generation': [],
            'mean_fitness': [],
            'best_fitness': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'gate_value': [],
        }

    def compute_fitness(
        self,
        params: np.ndarray,
        x: torch.Tensor,
        y_target: torch.Tensor
    ) -> float:
        """
        Compute fitness for a parameter vector on given batch.

        Fitness = negative cross-entropy loss (higher is better)
        """
        # Load parameters into model
        unflatten_params(self.model, params)
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(x)

            # Reshape for cross-entropy: (batch, num_digits+1, 10)
            num_output_digits = self.config.num_digits + 1
            y_pred = y_pred.view(-1, num_output_digits, 10)
            y_target_reshaped = y_target.view(-1, num_output_digits, 10)

            # Cross-entropy loss per digit
            loss = F.cross_entropy(
                y_pred.permute(0, 2, 1),  # (batch, 10, digits)
                y_target_reshaped.argmax(dim=2),    # (batch, digits)
                reduction='mean'
            )

        # Return negative loss as fitness (higher = better)
        return -loss.item()

    def evaluate(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate current model accuracy.

        Returns dict with accuracy metrics.
        """
        self.model.eval()

        x, y_target, target_ints = generate_addition_batch(
            num_samples, self.config.num_digits, self.device
        )

        with torch.no_grad():
            y_pred = self.model(x)

            # Decode predictions
            pred_ints = self.output_encoding.decode_batch(y_pred)

            # Exact match accuracy
            correct = (pred_ints == target_ints).float()
            accuracy = correct.mean().item()

            # Per-digit accuracy
            num_output_digits = self.config.num_digits + 1
            y_pred_digits = y_pred.view(-1, num_output_digits, 10).argmax(dim=2)
            y_target_digits = y_target.view(-1, num_output_digits, 10).argmax(dim=2)
            digit_accuracy = (y_pred_digits == y_target_digits).float().mean().item()

        results = {
            'accuracy': accuracy,
            'digit_accuracy': digit_accuracy,
        }

        # Track gate if available
        if hasattr(self.model, 'get_gate_value'):
            results['gate'] = self.model.get_gate_value()

        return results

    def evaluate_extrapolation(
        self,
        num_digits: int,
        num_samples: int = 500
    ) -> Dict[str, float]:
        """
        Evaluate on longer numbers than training.

        Note: This requires the model to handle different input sizes,
        which our current architecture doesn't support directly.
        For now, we test on the same size but could extend later.
        """
        # For initial experiments, just evaluate on training digit count
        # True extrapolation requires architectural changes
        return self.evaluate(num_samples)

    def train_generation(self) -> Dict[str, float]:
        """
        Run one generation of ES.

        Returns metrics for this generation.
        """
        # Generate a single batch for this generation (all members evaluated on same data)
        x, y_target, _ = generate_addition_batch(
            self.config.batch_size, self.config.num_digits, self.device
        )

        # Get perturbations
        epsilon, population = self.es.ask()

        # Evaluate each member on the SAME batch
        fitness = np.zeros(self.config.population_size)
        for i in range(self.config.population_size):
            fitness[i] = self.compute_fitness(population[i], x, y_target)

        # Update ES
        mean_fitness = self.es.tell(epsilon, fitness)

        # Load best params back into model
        unflatten_params(self.model, self.es.get_params())

        return {
            'mean_fitness': mean_fitness,
            'best_fitness': fitness.max(),
            'worst_fitness': fitness.min(),
        }

    def train(self, verbose: bool = True) -> Dict[str, list]:
        """
        Run full ES training.

        Returns training history.
        """
        if verbose:
            print(f"Starting ES training for {self.config.generations} generations")
            print(f"Population size: {self.config.population_size}")
            print(f"Parameters: {len(self.params):,}")
            print("-" * 60)

        start_time = time.time()

        for gen in range(self.config.generations):
            # Train one generation
            gen_metrics = self.train_generation()

            # Periodic evaluation
            if gen % self.config.eval_frequency == 0 or gen == self.config.generations - 1:
                eval_metrics = self.evaluate(self.config.eval_batch_size)

                self.history['generation'].append(gen)
                self.history['mean_fitness'].append(gen_metrics['mean_fitness'])
                self.history['best_fitness'].append(gen_metrics['best_fitness'])
                self.history['train_accuracy'].append(eval_metrics['accuracy'])
                self.history['test_accuracy'].append(eval_metrics['accuracy'])

                gate_val = eval_metrics.get('gate', None)
                self.history['gate_value'].append(gate_val)

                if verbose:
                    elapsed = time.time() - start_time
                    gate_str = f"gate={gate_val:.3f}" if gate_val is not None else ""
                    print(
                        f"Gen {gen:4d} | "
                        f"fitness={gen_metrics['mean_fitness']:.4f} | "
                        f"acc={eval_metrics['accuracy']:.4f} | "
                        f"digit_acc={eval_metrics['digit_accuracy']:.4f} | "
                        f"{gate_str} | "
                        f"time={elapsed:.1f}s"
                    )

            # Checkpointing
            if self.config.save_dir and gen % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(gen)

        # Final save
        if self.config.save_dir:
            self.save_checkpoint(self.config.generations, final=True)

        total_time = time.time() - start_time
        if verbose:
            print("-" * 60)
            print(f"Training complete in {total_time:.1f}s")
            print(f"Final accuracy: {self.history['train_accuracy'][-1]:.4f}")

        return self.history

    def save_checkpoint(self, generation: int, final: bool = False):
        """Save model checkpoint and history."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters
        suffix = 'final' if final else f'gen{generation:04d}'
        param_path = save_dir / f'params_{suffix}.npy'
        np.save(param_path, self.es.get_params())

        # Save history
        history_path = save_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, path: str):
        """Load model parameters from checkpoint."""
        params = np.load(path)
        self.es.set_params(params)
        unflatten_params(self.model, params)
