#!/usr/bin/env python3
"""
Experiment 006c: Start with 1-digit addition

The decoder needs to learn an autoencoder (input → hidden → input).
For 2-digit, this is 20 classes per operand. For 1-digit, it's only 10.
Test if the simpler task enables tool learning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from src.tools.arithmetic import AdditionTool
from src.dataset.addition import generate_addition_batch
from src.encodings.numbers import DigitEncoding


class ToolAugmentedMLP1Digit(nn.Module):
    """Simplified tool-augmented MLP for 1-digit addition."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.num_digits = 1
        self.encoding = DigitEncoding(1)
        self.tool = AdditionTool(max_digits=2)  # 1+1 can be 2 digits

        input_dim = 2 * 10  # Two 1-digit numbers, one-hot
        output_dim = 2 * 10  # Sum can be 0-18, encoded as 2 digits

        self.output_encoding = DigitEncoding(2)

        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Direct pathway
        self.direct_head = nn.Linear(hidden_dim, output_dim)

        # Tool pathway - simplified decoders
        # Each operand is just 10 classes (digits 0-9)
        self.decoder_a = nn.Linear(hidden_dim, 10)
        self.decoder_b = nn.Linear(hidden_dim, 10)

        # Gate
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        batch_size = x.shape[0]

        # Encode
        h = self.encoder(x)

        # Direct pathway
        direct_out = self.direct_head(h)

        # Tool pathway
        logits_a = self.decoder_a(h)  # (batch, 10)
        logits_b = self.decoder_b(h)  # (batch, 10)

        # Get discrete digits
        digit_a = logits_a.argmax(dim=1)  # (batch,)
        digit_b = logits_b.argmax(dim=1)  # (batch,)

        # Exact addition
        int_sum = self.tool.compute_batch(digit_a, digit_b)

        # Encode result
        sum_encoded = self.output_encoding.encode_batch(int_sum).to(x.device)

        # Gate and blend
        gate = torch.sigmoid(self.gate_logit)
        output = gate * sum_encoded + (1 - gate) * direct_out

        if return_gate:
            return output, gate
        return output

    def get_gate_value(self) -> float:
        return torch.sigmoid(self.gate_logit).item()

    def get_decoder_accuracy(self, x: torch.Tensor) -> Tuple[float, float]:
        """Check how well decoders reconstruct the original operands."""
        self.eval()
        with torch.no_grad():
            # Original operands (from input encoding)
            x_a = x[:, :10]  # First operand one-hot
            x_b = x[:, 10:]  # Second operand one-hot

            true_a = x_a.argmax(dim=1)
            true_b = x_b.argmax(dim=1)

            # Decoded operands
            h = self.encoder(x)
            pred_a = self.decoder_a(h).argmax(dim=1)
            pred_b = self.decoder_b(h).argmax(dim=1)

            acc_a = (pred_a == true_a).float().mean().item()
            acc_b = (pred_b == true_b).float().mean().item()

        return acc_a, acc_b


@dataclass
class GAConfig:
    population_size: int = 200
    elite_fraction: float = 0.1
    tournament_size: int = 5
    mutation_rate: float = 1.0
    mutation_std: float = 0.05
    seed: Optional[int] = None


def flatten_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])


def unflatten_params(model: nn.Module, params: np.ndarray):
    offset = 0
    for p in model.parameters():
        size = p.numel()
        p.data = torch.from_numpy(
            params[offset:offset + size].reshape(p.shape)
        ).float().to(p.device)
        offset += size


class GeneticAlgorithmTrainer:
    def __init__(self, model: nn.Module, config: GAConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        self.num_params = sum(p.numel() for p in model.parameters())
        base_params = flatten_params(model)

        self.population = np.zeros((config.population_size, self.num_params))
        for i in range(config.population_size):
            self.population[i] = base_params + np.random.randn(self.num_params) * 0.1

        self.best_params = base_params.copy()
        self.best_fitness = float('-inf')
        self.generation = 0

    def compute_fitness(self, params: np.ndarray, x: torch.Tensor, y_target: torch.Tensor) -> float:
        unflatten_params(self.model, params)
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(x)
            num_output_digits = 2  # 1-digit addition gives 2-digit output (0-18)

            y_pred = y_pred.view(-1, num_output_digits, 10)
            y_target_reshaped = y_target.view(-1, num_output_digits, 10)

            logits = y_pred.permute(0, 2, 1)
            labels = y_target_reshaped.argmax(dim=2)
            loss = F.cross_entropy(logits, labels, reduction='mean')

        return -loss.item()

    def evaluate_population(self, x: torch.Tensor, y_target: torch.Tensor) -> np.ndarray:
        fitness = np.zeros(self.config.population_size)
        for i in range(self.config.population_size):
            fitness[i] = self.compute_fitness(self.population[i], x, y_target)
        return fitness

    def tournament_select(self, fitness: np.ndarray) -> int:
        indices = np.random.choice(
            self.config.population_size,
            size=self.config.tournament_size,
            replace=False
        )
        return indices[np.argmax(fitness[indices])]

    def step(self, fitness: np.ndarray) -> float:
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_params = self.population[best_idx].copy()

        sorted_indices = np.argsort(fitness)[::-1]
        num_elite = max(1, int(self.config.population_size * self.config.elite_fraction))
        new_population = np.zeros_like(self.population)

        for i in range(num_elite):
            new_population[i] = self.population[sorted_indices[i]]

        for i in range(num_elite, self.config.population_size):
            parent_idx = self.tournament_select(fitness)
            child = self.population[parent_idx].copy()
            if np.random.random() < self.config.mutation_rate:
                child += np.random.randn(self.num_params) * self.config.mutation_std
            new_population[i] = child

        self.population = new_population
        self.generation += 1
        return float(np.mean(fitness))

    def evaluate_accuracy(self, x: torch.Tensor, y_target: torch.Tensor,
                         target_ints: torch.Tensor) -> dict:
        unflatten_params(self.model, self.best_params)
        self.model.eval()

        output_encoding = DigitEncoding(2)

        with torch.no_grad():
            y_pred, gate = self.model(x, return_gate=True)
            pred_ints = output_encoding.decode_batch(y_pred)

            exact_acc = (pred_ints == target_ints).float().mean().item()

            y_pred_digits = y_pred.view(-1, 2, 10).argmax(dim=2)
            y_target_digits = y_target.view(-1, 2, 10).argmax(dim=2)
            digit_acc = (y_pred_digits == y_target_digits).float().mean().item()

            # Decoder accuracy
            dec_a, dec_b = self.model.get_decoder_accuracy(x)

        return {
            'exact_acc': exact_acc,
            'digit_acc': digit_acc,
            'gate': gate.item(),
            'decoder_a': dec_a,
            'decoder_b': dec_b
        }


def run_experiment(hidden_dim: int, config: GAConfig, max_generations: int, device: str = 'cuda'):
    print(f"\nConfiguration:")
    print(f"  Task: 1-digit addition")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Population: {config.population_size}")
    print(f"  Mutation std: {config.mutation_std}")

    model = ToolAugmentedMLP1Digit(hidden_dim=hidden_dim)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    trainer = GeneticAlgorithmTrainer(model, config, device)

    # Generate evaluation data (1-digit addition)
    eval_x, eval_y, eval_ints = generate_addition_batch(2000, 1, device)

    batch_size = 256
    start_time = time.time()
    best_digit_acc = 0.0

    print(f"\nTraining for {max_generations} generations...")
    print("-" * 80)

    for gen in range(max_generations):
        x, y, _ = generate_addition_batch(batch_size, 1, device)
        fitness = trainer.evaluate_population(x, y)
        mean_fitness = trainer.step(fitness)

        if gen % 50 == 0 or gen == max_generations - 1:
            metrics = trainer.evaluate_accuracy(eval_x, eval_y, eval_ints)
            best_digit_acc = max(best_digit_acc, metrics['digit_acc'])
            elapsed = time.time() - start_time

            print(f"Gen {gen:5d} | fit={mean_fitness:+.4f} | "
                  f"exact={metrics['exact_acc']:.4f} | digit={metrics['digit_acc']:.4f} | "
                  f"gate={metrics['gate']:.3f} | dec_a={metrics['decoder_a']:.3f} | "
                  f"dec_b={metrics['decoder_b']:.3f} | {elapsed:.1f}s")

            if metrics['digit_acc'] >= 0.98:
                print(f"\nTarget reached at generation {gen}!")
                break

    final_metrics = trainer.evaluate_accuracy(eval_x, eval_y, eval_ints)
    return {
        **final_metrics,
        'best_digit_acc': best_digit_acc,
        'params': num_params,
        'generations': trainer.generation,
        'elapsed': time.time() - start_time
    }


def main():
    print("=" * 80)
    print("Experiment 006c: 1-Digit Addition (Simpler Decoder Task)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    configs = [
        # (hidden_dim, pop_size, mutation_std, max_gens)
        (32, 300, 0.05, 2000),
        (64, 500, 0.03, 3000),
    ]

    results = []

    for hidden_dim, pop_size, mut_std, max_gens in configs:
        print(f"\n{'='*80}")
        config = GAConfig(
            population_size=pop_size,
            elite_fraction=0.1,
            tournament_size=5,
            mutation_rate=1.0,
            mutation_std=mut_std,
            seed=42
        )

        result = run_experiment(hidden_dim, config, max_gens, device)
        result['hidden_dim'] = hidden_dim
        results.append(result)

        status = "PASS" if result['digit_acc'] >= 0.95 else \
                 ("PARTIAL" if result['digit_acc'] >= 0.7 else "FAIL")
        print(f"\nResult: digit_acc={result['digit_acc']:.4f}, gate={result['gate']:.3f}, "
              f"decoder_a={result['decoder_a']:.3f}, decoder_b={result['decoder_b']:.3f} - {status}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary (1-Digit Addition)")
    print("=" * 80)
    print(f"{'Hidden':>7} {'Params':>8} {'Digit':>8} {'Exact':>8} {'Gate':>6} {'Dec_A':>7} {'Dec_B':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['hidden_dim']:>7} {r['params']:>8} {r['digit_acc']:>8.4f} "
              f"{r['exact_acc']:>8.4f} {r['gate']:>6.3f} {r['decoder_a']:>7.3f} {r['decoder_b']:>7.3f}")


if __name__ == '__main__':
    main()
