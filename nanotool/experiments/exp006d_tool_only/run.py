#!/usr/bin/env python3
"""
Experiment 006d: Tool-Only Architecture (No Direct Pathway)

Force the network to use the tool by removing the direct pathway.
The network MUST learn to: encode → decode operands → tool → encode result.
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


class ToolOnlyMLP(nn.Module):
    """
    MLP that MUST use the tool - no direct pathway.

    Architecture:
    1. Encoder: (a, b) → hidden
    2. Decoder: hidden → (a', b') operand logits
    3. Tool: argmax(a'), argmax(b') → exact_add → sum
    4. Output encoder: sum_onehot → output
    """

    def __init__(self, num_digits: int, hidden_dim: int = 64):
        super().__init__()

        self.num_digits = num_digits
        self.encoding = DigitEncoding(num_digits)
        self.output_encoding = DigitEncoding(num_digits + 1)
        self.tool = AdditionTool(max_digits=num_digits + 1)

        input_dim = 2 * num_digits * 10
        output_dim = (num_digits + 1) * 10

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Decoders for each operand
        self.decoder_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_digits * 10)
        )
        self.decoder_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_digits * 10)
        )

        # Output projection (tool result → final output)
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # Encode
        h = self.encoder(x)

        # Decode operands
        logits_a = self.decoder_a(h).view(batch_size, self.num_digits, 10)
        logits_b = self.decoder_b(h).view(batch_size, self.num_digits, 10)

        # Get discrete digits
        digits_a = logits_a.argmax(dim=2)  # (batch, num_digits)
        digits_b = logits_b.argmax(dim=2)

        # Convert to integers
        multipliers = torch.tensor(
            [10 ** (self.num_digits - 1 - i) for i in range(self.num_digits)],
            device=x.device
        ).float()

        int_a = (digits_a.float() * multipliers).sum(dim=1).long()
        int_b = (digits_b.float() * multipliers).sum(dim=1).long()

        # Exact addition via tool
        int_sum = self.tool.compute_batch(int_a, int_b)

        # Encode result
        sum_encoded = self.output_encoding.encode_batch(int_sum).to(x.device)

        # Project to output
        output = self.output_proj(sum_encoded)

        return output

    def get_decoder_accuracy(self, x: torch.Tensor) -> Tuple[float, float]:
        """Check decoder reconstruction accuracy."""
        self.eval()
        batch_size = x.shape[0]
        single_dim = self.num_digits * 10

        with torch.no_grad():
            # True operands
            x_a = x[:, :single_dim].view(batch_size, self.num_digits, 10)
            x_b = x[:, single_dim:].view(batch_size, self.num_digits, 10)
            true_a = x_a.argmax(dim=2)
            true_b = x_b.argmax(dim=2)

            # Decoded operands
            h = self.encoder(x)
            pred_a = self.decoder_a(h).view(batch_size, self.num_digits, 10).argmax(dim=2)
            pred_b = self.decoder_b(h).view(batch_size, self.num_digits, 10).argmax(dim=2)

            # Digit-level accuracy
            acc_a = (pred_a == true_a).float().mean().item()
            acc_b = (pred_b == true_b).float().mean().item()

        return acc_a, acc_b


@dataclass
class GAConfig:
    population_size: int = 500
    elite_fraction: float = 0.1
    tournament_size: int = 5
    mutation_rate: float = 1.0
    mutation_std: float = 0.03
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

    def compute_fitness(self, params: np.ndarray, x: torch.Tensor, y_target: torch.Tensor) -> float:
        unflatten_params(self.model, params)
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(x)
            num_output_digits = y_target.shape[1] // 10

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
        indices = np.random.choice(len(fitness), size=self.config.tournament_size, replace=False)
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
        return float(np.mean(fitness))

    def evaluate_accuracy(self, x: torch.Tensor, y_target: torch.Tensor,
                         target_ints: torch.Tensor, num_digits: int) -> dict:
        unflatten_params(self.model, self.best_params)
        self.model.eval()

        output_encoding = DigitEncoding(num_digits + 1)

        with torch.no_grad():
            y_pred = self.model(x)
            pred_ints = output_encoding.decode_batch(y_pred)

            exact_acc = (pred_ints == target_ints).float().mean().item()

            num_output_digits = num_digits + 1
            y_pred_digits = y_pred.view(-1, num_output_digits, 10).argmax(dim=2)
            y_target_digits = y_target.view(-1, num_output_digits, 10).argmax(dim=2)
            digit_acc = (y_pred_digits == y_target_digits).float().mean().item()

            dec_a, dec_b = self.model.get_decoder_accuracy(x)

        return {
            'exact_acc': exact_acc,
            'digit_acc': digit_acc,
            'decoder_a': dec_a,
            'decoder_b': dec_b
        }


def run_experiment(num_digits: int, hidden_dim: int, config: GAConfig,
                   max_generations: int, device: str = 'cuda'):
    print(f"\nConfiguration:")
    print(f"  Task: {num_digits}-digit addition (TOOL ONLY)")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Population: {config.population_size}")
    print(f"  Mutation std: {config.mutation_std}")

    model = ToolOnlyMLP(num_digits=num_digits, hidden_dim=hidden_dim)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    trainer = GeneticAlgorithmTrainer(model, config, device)
    eval_x, eval_y, eval_ints = generate_addition_batch(2000, num_digits, device)

    batch_size = 256
    start_time = time.time()
    best_digit_acc = 0.0

    print(f"\nTraining for {max_generations} generations...")
    print("-" * 80)

    for gen in range(max_generations):
        x, y, _ = generate_addition_batch(batch_size, num_digits, device)
        fitness = trainer.evaluate_population(x, y)
        mean_fitness = trainer.step(fitness)

        if gen % 100 == 0 or gen == max_generations - 1:
            metrics = trainer.evaluate_accuracy(eval_x, eval_y, eval_ints, num_digits)
            best_digit_acc = max(best_digit_acc, metrics['digit_acc'])
            elapsed = time.time() - start_time

            print(f"Gen {gen:5d} | fit={mean_fitness:+.4f} | "
                  f"exact={metrics['exact_acc']:.4f} | digit={metrics['digit_acc']:.4f} | "
                  f"dec_a={metrics['decoder_a']:.3f} | dec_b={metrics['decoder_b']:.3f} | {elapsed:.1f}s")

            if metrics['digit_acc'] >= 0.95:
                print(f"\nTarget reached at generation {gen}!")
                break

    final_metrics = trainer.evaluate_accuracy(eval_x, eval_y, eval_ints, num_digits)
    return {
        **final_metrics,
        'best_digit_acc': best_digit_acc,
        'params': num_params,
        'elapsed': time.time() - start_time
    }


def main():
    print("=" * 80)
    print("Experiment 006d: Tool-Only Architecture (Forced Tool Use)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Start simple: 1-digit addition
    configs = [
        # (num_digits, hidden_dim, pop_size, mutation_std, max_gens)
        (1, 32, 500, 0.05, 3000),
        (1, 64, 1000, 0.03, 5000),
    ]

    results = []

    for num_digits, hidden_dim, pop_size, mut_std, max_gens in configs:
        print(f"\n{'='*80}")
        config = GAConfig(
            population_size=pop_size,
            elite_fraction=0.1,
            tournament_size=5,
            mutation_rate=1.0,
            mutation_std=mut_std,
            seed=42
        )

        result = run_experiment(num_digits, hidden_dim, config, max_gens, device)
        result['num_digits'] = num_digits
        result['hidden_dim'] = hidden_dim
        results.append(result)

        status = "PASS" if result['digit_acc'] >= 0.95 else \
                 ("PARTIAL" if result['digit_acc'] >= 0.7 else "FAIL")
        print(f"\nResult: digit_acc={result['digit_acc']:.4f}, "
              f"decoder_a={result['decoder_a']:.3f}, decoder_b={result['decoder_b']:.3f} - {status}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary (Tool-Only Architecture)")
    print("=" * 80)
    print(f"{'Digits':>6} {'Hidden':>7} {'Params':>8} {'Digit':>8} {'Exact':>8} {'Dec_A':>7} {'Dec_B':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['num_digits']:>6} {r['hidden_dim']:>7} {r['params']:>8} "
              f"{r['digit_acc']:>8.4f} {r['exact_acc']:>8.4f} {r['decoder_a']:>7.3f} {r['decoder_b']:>7.3f}")


if __name__ == '__main__':
    main()
