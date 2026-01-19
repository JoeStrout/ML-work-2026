#!/usr/bin/env python3
"""
Experiment 004: GA with Batched GPU Evaluation

Test genetic algorithm on autoencoder task with entire population
evaluated in parallel on GPU.
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
    """Generate batch of one-hot encoded numbers."""
    encoding = DigitEncoding(num_digits)
    numbers = torch.randint(0, 10 ** num_digits, (batch_size,))
    x = encoding.encode_batch(numbers).to(device)
    return x, numbers


def main():
    print("=" * 60)
    print("Experiment 004: GA with Batched GPU Evaluation")
    print("=" * 60)

    # Configuration
    num_digits = 4
    hidden_dim = 64
    batch_size = 512
    population_size = 500  # Large population, evaluated in parallel!
    max_generations = 2000
    eval_frequency = 50
    target_accuracy = 0.95

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Model architecture
    input_dim = num_digits * 10
    layer_sizes = [input_dim, hidden_dim, input_dim]

    print(f"Architecture: {layer_sizes}")
    print(f"Population size: {population_size}")

    # Create batched model
    model = BatchedMLP(
        layer_sizes=layer_sizes,
        population_size=population_size,
        activation='silu',
        device=device
    )
    print(f"Parameters per individual: {model.num_params_per_member:,}")
    print(f"Total parameters (all population): {model.num_params_per_member * population_size:,}")

    # GA configuration
    config = GAConfig(
        population_size=population_size,
        elite_fraction=0.1,
        tournament_size=5,
        mutation_rate=1.0,
        mutation_std=0.1,
        crossover_rate=0.0,  # Often harmful for NN weights
        seed=42
    )

    ga = GeneticAlgorithm(config, model)
    fitness_fn = create_autoencoder_fitness_fn(num_digits)

    print(f"\nGA Config:")
    print(f"  Elite fraction: {config.elite_fraction}")
    print(f"  Tournament size: {config.tournament_size}")
    print(f"  Mutation std: {config.mutation_std}")

    print(f"\nTraining for up to {max_generations} generations")
    print("-" * 60)

    # Fixed evaluation batch for consistent accuracy measurement
    eval_x, _ = generate_batch(2000, num_digits, device)

    start_time = time.time()
    best_accuracy = 0.0

    for gen in range(max_generations):
        # Generate training batch
        x, _ = generate_batch(batch_size, num_digits, device)

        # Evaluate fitness (all population members in parallel)
        fitness = ga.evaluate(x, fitness_fn)

        # GA step
        mean_fitness = ga.step(fitness)

        # Periodic evaluation
        if gen % eval_frequency == 0:
            with torch.no_grad():
                outputs = model(eval_x)
                accuracies = compute_batch_accuracy(outputs, eval_x, num_digits)
                best_acc_this_gen = accuracies.max().item()
                mean_acc = accuracies.mean().item()

            best_accuracy = max(best_accuracy, best_acc_this_gen)
            elapsed = time.time() - start_time

            print(
                f"Gen {gen:5d} | "
                f"fit={mean_fitness:.6f} | "
                f"acc_best={best_acc_this_gen:.4f} | "
                f"acc_mean={mean_acc:.4f} | "
                f"best_ever={best_accuracy:.4f} | "
                f"{elapsed:.1f}s"
            )

            if best_acc_this_gen >= target_accuracy:
                print(f"\nTarget reached!")
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    with torch.no_grad():
        # Use larger batch for final eval
        final_x, _ = generate_batch(5000, num_digits, device)
        outputs = model(final_x)
        accuracies = compute_batch_accuracy(outputs, final_x, num_digits)

        best_idx = accuracies.argmax().item()
        best_acc = accuracies[best_idx].item()
        mean_acc = accuracies.mean().item()

    total_time = time.time() - start_time
    print(f"Best individual accuracy: {best_acc:.4f}")
    print(f"Population mean accuracy: {mean_acc:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Generations per second: {gen / total_time:.1f}")

    if best_acc >= 0.95:
        print("\nSUCCESS!")
    elif best_acc >= 0.8:
        print("\nPARTIAL SUCCESS")
    else:
        print("\nFAILURE")


if __name__ == '__main__':
    main()
