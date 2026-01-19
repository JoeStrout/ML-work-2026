#!/usr/bin/env python3
"""
Experiment 006b: Tool-Augmented Addition with Skip Connections

The original architecture struggles because the decoder must recover both
operands from a shared hidden representation. Add skip connections to
provide the decoder with direct access to the original encoded inputs.
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


class ToolAugmentedMLPWithSkip(nn.Module):
    """
    Tool-augmented MLP with skip connections to decoders.

    Key change: The tool decoders receive both the hidden representation
    AND the original encoded inputs, making it easier to recover operands.
    """

    def __init__(
        self,
        num_digits: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        self.num_digits = num_digits
        self.encoding = DigitEncoding(num_digits)
        self.tool = AdditionTool(max_digits=num_digits + 1)

        input_dim = 2 * self.encoding.encoding_dim  # Concatenated a, b
        single_input_dim = self.encoding.encoding_dim  # Single operand
        output_dim = DigitEncoding(num_digits + 1).encoding_dim

        self.output_encoding = DigitEncoding(num_digits + 1)

        # Encoder network
        encoder_layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_hidden_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.encoder = nn.ModuleList(encoder_layers)

        # Direct pathway (bypasses tool)
        self.direct_head = nn.Linear(hidden_dim, output_dim)

        # Tool pathway decoders WITH skip connections
        # Each decoder sees: hidden + original encoded input for that operand
        decoder_input_dim = hidden_dim + single_input_dim

        self.tool_decoder_a = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, single_input_dim)
        )
        self.tool_decoder_b = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, single_input_dim)
        )

        # Tool output encoder
        self.tool_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Gate
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        batch_size = x.shape[0]
        single_dim = self.encoding.encoding_dim

        # Split original input for skip connections
        x_a = x[:, :single_dim]  # First operand encoding
        x_b = x[:, single_dim:]  # Second operand encoding

        # Encode through shared encoder
        h = x
        for layer in self.encoder:
            h = F.silu(layer(h))

        # === Direct pathway ===
        direct_out = self.direct_head(h)

        # === Tool pathway with skip connections ===
        # Concatenate hidden with original input for each decoder
        decoder_input_a = torch.cat([h, x_a], dim=1)
        decoder_input_b = torch.cat([h, x_b], dim=1)

        logits_a = self.tool_decoder_a(decoder_input_a)
        logits_b = self.tool_decoder_b(decoder_input_b)

        # Reshape and get discrete digits
        logits_a = logits_a.view(batch_size, self.num_digits, -1)
        logits_b = logits_b.view(batch_size, self.num_digits, -1)

        digits_a = logits_a.argmax(dim=2)
        digits_b = logits_b.argmax(dim=2)

        # Convert to integers
        base = 10
        multipliers = torch.tensor(
            [base ** (self.num_digits - 1 - i) for i in range(self.num_digits)],
            device=x.device
        ).float()

        int_a = (digits_a.float() * multipliers).sum(dim=1).long()
        int_b = (digits_b.float() * multipliers).sum(dim=1).long()

        # Exact addition via tool
        int_sum = self.tool.compute_batch(int_a, int_b)

        # Encode result
        sum_encoded = self.output_encoding.encode_batch(int_sum).to(x.device)
        tool_out = self.tool_encoder(sum_encoded)

        # === Gate and blend ===
        gate = torch.sigmoid(self.gate_logit)
        output = gate * tool_out + (1 - gate) * direct_out

        if return_gate:
            return output, gate
        return output

    def get_gate_value(self) -> float:
        return torch.sigmoid(self.gate_logit).item()


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
    """GA trainer with cross-entropy fitness."""

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
                         target_ints: torch.Tensor) -> Tuple[float, float, float]:
        unflatten_params(self.model, self.best_params)
        self.model.eval()

        num_digits = self.model.num_digits
        output_encoding = DigitEncoding(num_digits + 1)

        with torch.no_grad():
            y_pred, gate = self.model(x, return_gate=True)
            pred_ints = output_encoding.decode_batch(y_pred)

            exact_acc = (pred_ints == target_ints).float().mean().item()

            num_output_digits = num_digits + 1
            y_pred_digits = y_pred.view(-1, num_output_digits, 10).argmax(dim=2)
            y_target_digits = y_target.view(-1, num_output_digits, 10).argmax(dim=2)
            digit_acc = (y_pred_digits == y_target_digits).float().mean().item()

            gate_val = gate.item()

        return exact_acc, digit_acc, gate_val


def run_experiment(num_digits: int, hidden_dim: int, config: GAConfig,
                   max_generations: int, device: str = 'cuda'):
    print(f"\nConfiguration:")
    print(f"  Digits: {num_digits}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Population: {config.population_size}")
    print(f"  Mutation std: {config.mutation_std}")

    model = ToolAugmentedMLPWithSkip(
        num_digits=num_digits,
        hidden_dim=hidden_dim,
        num_hidden_layers=2
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    trainer = GeneticAlgorithmTrainer(model, config, device)
    eval_x, eval_y, eval_ints = generate_addition_batch(2000, num_digits, device)

    batch_size = 256
    start_time = time.time()
    best_digit_acc = 0.0

    print(f"\nTraining for {max_generations} generations...")
    print("-" * 70)

    for gen in range(max_generations):
        x, y, _ = generate_addition_batch(batch_size, num_digits, device)
        fitness = trainer.evaluate_population(x, y)
        mean_fitness = trainer.step(fitness)

        if gen % 50 == 0 or gen == max_generations - 1:
            exact_acc, digit_acc, gate = trainer.evaluate_accuracy(eval_x, eval_y, eval_ints)
            best_digit_acc = max(best_digit_acc, digit_acc)
            elapsed = time.time() - start_time

            print(f"Gen {gen:5d} | fit={mean_fitness:+.4f} | "
                  f"exact={exact_acc:.4f} | digit={digit_acc:.4f} | "
                  f"gate={gate:.3f} | best={best_digit_acc:.4f} | {elapsed:.1f}s")

            if digit_acc >= 0.95:
                print(f"\nTarget reached at generation {gen}!")
                break

    exact_acc, digit_acc, gate = trainer.evaluate_accuracy(eval_x, eval_y, eval_ints)

    return {
        'exact_accuracy': exact_acc,
        'digit_accuracy': digit_acc,
        'gate': gate,
        'best_digit_accuracy': best_digit_acc,
        'generations': trainer.generation,
        'params': num_params,
        'elapsed': time.time() - start_time
    }


def main():
    print("=" * 70)
    print("Experiment 006b: Tool-Augmented with Skip Connections")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test configurations
    configs = [
        # (num_digits, hidden_dim, pop_size, mutation_std, max_gens)
        (2, 32, 300, 0.05, 2000),
        (2, 64, 500, 0.03, 3000),
    ]

    results = []

    for num_digits, hidden_dim, pop_size, mut_std, max_gens in configs:
        print(f"\n{'='*70}")
        print(f"Test: {num_digits}-digit addition with skip connections")
        print(f"{'='*70}")

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

        status = "PASS" if result['digit_accuracy'] >= 0.9 else \
                 ("PARTIAL" if result['digit_accuracy'] >= 0.5 else "FAIL")
        print(f"\nResult: digit_acc={result['digit_accuracy']:.4f}, "
              f"exact_acc={result['exact_accuracy']:.4f}, gate={result['gate']:.3f} - {status}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary (Skip Connections)")
    print("=" * 70)
    print(f"{'Digits':>6} {'Hidden':>7} {'Params':>8} {'Digit Acc':>10} {'Exact Acc':>10} {'Gate':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['num_digits']:>6} {r['hidden_dim']:>7} {r['params']:>8} "
              f"{r['digit_accuracy']:>10.4f} {r['exact_accuracy']:>10.4f} {r['gate']:>6.3f}")


if __name__ == '__main__':
    main()
