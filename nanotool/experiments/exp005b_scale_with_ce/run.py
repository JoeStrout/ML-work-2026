#!/usr/bin/env python3
"""
Experiment 005b: Test if cross-entropy fitness scales to larger problems.

Now that we know CE dramatically outperforms MSE, test on 4-digit (5224 params)
which previously failed with MSE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np
import time

from src.evolution.ga import (
    GeneticAlgorithm, BatchedMLP, GAConfig,
    compute_batch_accuracy
)
from src.encodings.numbers import DigitEncoding


def generate_batch(batch_size: int, num_digits: int, device: str = 'cuda'):
    encoding = DigitEncoding(num_digits)
    numbers = torch.randint(0, 10 ** num_digits, (batch_size,))
    x = encoding.encode_batch(numbers).to(device)
    return x, numbers


def fitness_cross_entropy(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """Per-digit cross-entropy loss."""
    pop_size, batch_size, _ = outputs.shape
    outputs_reshaped = outputs.view(pop_size, batch_size, num_digits, 10)
    x_reshaped = x.view(batch_size, num_digits, 10)
    targets = x_reshaped.argmax(dim=2)

    ce_loss = torch.zeros(pop_size, device=outputs.device)
    for p in range(pop_size):
        logits = outputs_reshaped[p].view(-1, 10)
        labels = targets.view(-1)
        ce_loss[p] = F.cross_entropy(logits, labels, reduction='mean')

    return -ce_loss


def run_experiment(num_digits, hidden_dim, population_size, mutation_std, max_gens):
    """Run GA with cross-entropy fitness."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = num_digits * 10
    layer_sizes = [input_dim, hidden_dim, input_dim]

    model = BatchedMLP(
        layer_sizes=layer_sizes,
        population_size=population_size,
        activation='silu',
        device=device
    )

    config = GAConfig(
        population_size=population_size,
        elite_fraction=0.1,
        tournament_size=5,
        mutation_rate=1.0,
        mutation_std=mutation_std,
        crossover_rate=0.0,
        seed=42
    )

    ga = GeneticAlgorithm(config, model)

    def wrapped_fitness(outputs, x):
        return fitness_cross_entropy(outputs, x, num_digits)

    eval_x, _ = generate_batch(2000, num_digits, device)
    batch_size = 512

    start_time = time.time()
    best_accuracy = 0.0

    for gen in range(max_gens):
        x, _ = generate_batch(batch_size, num_digits, device)
        fitness = ga.evaluate(x, wrapped_fitness)
        mean_fit = ga.step(fitness)

        if gen % 100 == 0 or gen == max_gens - 1:
            with torch.no_grad():
                outputs = model(eval_x)
                accuracies = compute_batch_accuracy(outputs, eval_x, num_digits)
                best_acc = accuracies.max().item()
                mean_acc = accuracies.mean().item()

            best_accuracy = max(best_accuracy, best_acc)
            elapsed = time.time() - start_time

            print(f"  Gen {gen:5d} | best={best_acc:.4f} | mean={mean_acc:.4f} | "
                  f"fit={mean_fit:.4f} | {elapsed:.1f}s")

            if best_acc >= 0.95:
                print(f"\n  Target reached at generation {gen}!")
                break

    return best_accuracy, model.num_params_per_member, time.time() - start_time


def main():
    print("=" * 70)
    print("Experiment 005b: Scaling with Cross-Entropy Fitness")
    print("=" * 70)

    configs = [
        # (num_digits, hidden_dim, pop_size, mutation_std, max_gens)
        (2, 32, 500, 0.1, 1000),     # 1,332 params - should work
        (3, 32, 500, 0.1, 2000),     # 2,048 params
        (4, 64, 500, 0.1, 3000),     # 5,224 params - previously failed
        (4, 64, 1000, 0.05, 5000),   # 5,224 params - larger pop, lower mutation
    ]

    results = []

    for num_digits, hidden_dim, pop_size, mut_std, max_gens in configs:
        input_dim = num_digits * 10
        est_params = input_dim * hidden_dim + hidden_dim + hidden_dim * input_dim + input_dim

        print(f"\n{'='*70}")
        print(f"{num_digits} digits, hidden={hidden_dim}, pop={pop_size}, mut={mut_std}")
        print(f"Estimated params: {est_params}")
        print(f"{'='*70}")

        best_acc, params, elapsed = run_experiment(
            num_digits, hidden_dim, pop_size, mut_std, max_gens
        )

        status = "PASS" if best_acc >= 0.9 else ("PARTIAL" if best_acc >= 0.5 else "FAIL")
        results.append((num_digits, hidden_dim, pop_size, mut_std, params, best_acc, elapsed, status))
        print(f"\nResult: {best_acc:.4f} - {status}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary (Cross-Entropy Fitness)")
    print("=" * 70)
    print(f"{'Digits':>6} {'Hidden':>7} {'Pop':>5} {'Mut':>5} {'Params':>7} {'Acc':>8} {'Time':>8} {'Status':>8}")
    print("-" * 70)
    for d, h, p, m, params, acc, elapsed, status in results:
        print(f"{d:>6} {h:>7} {p:>5} {m:>5.2f} {params:>7} {acc:>8.4f} {elapsed:>7.1f}s {status:>8}")


if __name__ == '__main__':
    main()
