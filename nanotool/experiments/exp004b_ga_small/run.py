#!/usr/bin/env python3
"""
Experiment 004b: GA on smaller networks

Test if GA can learn when we reduce parameters to the ~400 range
where we know ES succeeded.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import time

from src.evolution.ga import (
    GeneticAlgorithm, BatchedMLP, GAConfig,
    create_autoencoder_fitness_fn, compute_batch_accuracy
)
from src.encodings.numbers import DigitEncoding


def generate_batch(batch_size: int, num_digits: int, device: str = 'cuda'):
    encoding = DigitEncoding(num_digits)
    numbers = torch.randint(0, 10 ** num_digits, (batch_size,))
    x = encoding.encode_batch(numbers).to(device)
    return x, numbers


def run_experiment(num_digits, hidden_dim, population_size, mutation_std, max_gens=3000):
    """Run GA experiment with given configuration."""
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
    fitness_fn = create_autoencoder_fitness_fn(num_digits)

    eval_x, _ = generate_batch(2000, num_digits, device)
    batch_size = 512

    start_time = time.time()
    best_accuracy = 0.0

    for gen in range(max_gens):
        x, _ = generate_batch(batch_size, num_digits, device)
        fitness = ga.evaluate(x, fitness_fn)
        ga.step(fitness)

        if gen % 200 == 0 or gen == max_gens - 1:
            with torch.no_grad():
                outputs = model(eval_x)
                accuracies = compute_batch_accuracy(outputs, eval_x, num_digits)
                best_acc = accuracies.max().item()
                mean_acc = accuracies.mean().item()

            best_accuracy = max(best_accuracy, best_acc)
            elapsed = time.time() - start_time
            print(f"  Gen {gen:5d} | best={best_acc:.4f} | mean={mean_acc:.4f} | {elapsed:.1f}s")

            if best_acc >= 0.95:
                break

    return best_accuracy, model.num_params_per_member, time.time() - start_time


def main():
    print("=" * 60)
    print("Experiment 004b: GA on Smaller Networks")
    print("=" * 60)

    configs = [
        # (num_digits, hidden_dim, pop_size, mutation_std)
        (1, 20, 200, 0.1),   # ~430 params - should work
        (1, 20, 500, 0.05),  # ~430 params - larger pop, smaller mutation
        (2, 16, 500, 0.1),   # ~550 params
        (2, 32, 500, 0.1),   # ~1100 params
        (3, 16, 500, 0.1),   # ~800 params
        (4, 16, 1000, 0.05), # ~1050 params - more pop, less mutation
    ]

    results = []

    for num_digits, hidden_dim, pop_size, mut_std in configs:
        input_dim = num_digits * 10
        est_params = input_dim * hidden_dim + hidden_dim + hidden_dim * input_dim + input_dim

        print(f"\n{num_digits} digits, hidden={hidden_dim}, pop={pop_size}, mut={mut_std}")
        print(f"  Estimated params: {est_params}")

        best_acc, params, elapsed = run_experiment(
            num_digits, hidden_dim, pop_size, mut_std, max_gens=3000
        )

        status = "PASS" if best_acc >= 0.9 else "FAIL"
        results.append((num_digits, hidden_dim, pop_size, mut_std, params, best_acc, status))
        print(f"  Final: {best_acc:.4f} - {status}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Digits':>6} {'Hidden':>7} {'Pop':>5} {'Mut':>5} {'Params':>7} {'Acc':>7} {'Status':>6}")
    print("-" * 55)
    for d, h, p, m, params, acc, status in results:
        print(f"{d:>6} {h:>7} {p:>5} {m:>5.2f} {params:>7} {acc:>7.4f} {status:>6}")


if __name__ == '__main__':
    main()
