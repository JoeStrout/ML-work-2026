#!/usr/bin/env python3
"""
Experiment 005: Compare different fitness functions for GA.

Test whether alternative fitness formulations provide more tractable
optimization landscapes for evolutionary methods.
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
    """Generate batch of one-hot encoded numbers."""
    encoding = DigitEncoding(num_digits)
    numbers = torch.randint(0, 10 ** num_digits, (batch_size,))
    x = encoding.encode_batch(numbers).to(device)
    return x, numbers


# ============================================================================
# Different fitness functions
# ============================================================================

def fitness_mse(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """Original: Negative MSE on raw outputs."""
    x_expanded = x.unsqueeze(0).expand_as(outputs)
    mse = ((outputs - x_expanded) ** 2).mean(dim=(1, 2))
    return -mse


def fitness_cross_entropy(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """Per-digit cross-entropy loss."""
    pop_size, batch_size, _ = outputs.shape

    # Reshape to (pop, batch, digits, 10)
    outputs_reshaped = outputs.view(pop_size, batch_size, num_digits, 10)
    x_reshaped = x.view(batch_size, num_digits, 10)

    # Get target class indices
    targets = x_reshaped.argmax(dim=2)  # (batch, digits)

    # Compute cross-entropy per population member
    # Need to iterate because cross_entropy doesn't handle batched weights well
    ce_loss = torch.zeros(pop_size, device=outputs.device)
    for p in range(pop_size):
        # outputs_reshaped[p]: (batch, digits, 10)
        # Reshape for cross_entropy: (batch*digits, 10)
        logits = outputs_reshaped[p].view(-1, 10)
        labels = targets.view(-1)
        ce_loss[p] = F.cross_entropy(logits, labels, reduction='mean')

    return -ce_loss  # Negative because higher is better


def fitness_log_softmax(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """Sum of log-softmax probabilities for correct classes."""
    pop_size, batch_size, _ = outputs.shape

    # Reshape to (pop, batch, digits, 10)
    outputs_reshaped = outputs.view(pop_size, batch_size, num_digits, 10)
    x_reshaped = x.view(batch_size, num_digits, 10)

    # Get target class indices
    targets = x_reshaped.argmax(dim=2)  # (batch, digits)
    targets_expanded = targets.unsqueeze(0).expand(pop_size, -1, -1)  # (pop, batch, digits)

    # Log-softmax over class dimension
    log_probs = F.log_softmax(outputs_reshaped, dim=3)  # (pop, batch, digits, 10)

    # Gather log-probs for correct classes
    correct_log_probs = log_probs.gather(3, targets_expanded.unsqueeze(3)).squeeze(3)

    # Sum over digits, mean over batch
    fitness = correct_log_probs.sum(dim=2).mean(dim=1)

    return fitness


def fitness_digit_accuracy(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """Discrete digit accuracy (count of correct digits)."""
    return compute_batch_accuracy(outputs, x, num_digits)


def fitness_soft_accuracy(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """
    Soft accuracy: softmax probability mass on correct classes.

    More gradient-friendly than discrete accuracy.
    """
    pop_size, batch_size, _ = outputs.shape

    # Reshape to (pop, batch, digits, 10)
    outputs_reshaped = outputs.view(pop_size, batch_size, num_digits, 10)
    x_reshaped = x.view(batch_size, num_digits, 10)

    # Softmax over classes
    probs = F.softmax(outputs_reshaped, dim=3)  # (pop, batch, digits, 10)

    # Get target class indices
    targets = x_reshaped.argmax(dim=2)  # (batch, digits)
    targets_expanded = targets.unsqueeze(0).expand(pop_size, -1, -1)  # (pop, batch, digits)

    # Gather probability for correct classes
    correct_probs = probs.gather(3, targets_expanded.unsqueeze(3)).squeeze(3)

    # Mean probability of correct class
    fitness = correct_probs.mean(dim=(1, 2))

    return fitness


def fitness_rank_based(outputs: torch.Tensor, x: torch.Tensor, num_digits: int) -> torch.Tensor:
    """
    Rank-based fitness: reward based on rank of correct class.

    Top-1 = 1.0, Top-2 = 0.9, ..., Bottom = 0.0
    """
    pop_size, batch_size, _ = outputs.shape

    # Reshape to (pop, batch, digits, 10)
    outputs_reshaped = outputs.view(pop_size, batch_size, num_digits, 10)
    x_reshaped = x.view(batch_size, num_digits, 10)

    # Get target class indices
    targets = x_reshaped.argmax(dim=2)  # (batch, digits)

    # Compute ranks
    # argsort gives indices that would sort; argsort again gives ranks
    sorted_indices = outputs_reshaped.argsort(dim=3, descending=True)
    ranks = sorted_indices.argsort(dim=3)  # rank 0 = highest

    # Gather rank of correct class
    targets_expanded = targets.unsqueeze(0).expand(pop_size, -1, -1).unsqueeze(3)
    correct_ranks = ranks.gather(3, targets_expanded).squeeze(3).float()

    # Convert rank to score: rank 0 -> 1.0, rank 9 -> 0.0
    scores = 1.0 - (correct_ranks / 9.0)

    return scores.mean(dim=(1, 2))


# ============================================================================
# Experiment runner
# ============================================================================

def run_experiment(fitness_fn, fitness_name, num_digits, hidden_dim,
                   population_size, mutation_std, max_gens=2000):
    """Run GA with given fitness function."""
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

    # Wrapper to pass num_digits
    def wrapped_fitness(outputs, x):
        return fitness_fn(outputs, x, num_digits)

    eval_x, _ = generate_batch(2000, num_digits, device)
    batch_size = 512

    start_time = time.time()
    best_accuracy = 0.0
    history = []

    for gen in range(max_gens):
        x, _ = generate_batch(batch_size, num_digits, device)
        fitness = ga.evaluate(x, wrapped_fitness)
        mean_fit = ga.step(fitness)

        if gen % 200 == 0 or gen == max_gens - 1:
            with torch.no_grad():
                outputs = model(eval_x)
                accuracies = compute_batch_accuracy(outputs, eval_x, num_digits)
                best_acc = accuracies.max().item()
                mean_acc = accuracies.mean().item()

            best_accuracy = max(best_accuracy, best_acc)
            elapsed = time.time() - start_time
            history.append((gen, best_acc, mean_acc, mean_fit))

            print(f"  Gen {gen:5d} | best={best_acc:.4f} | mean={mean_acc:.4f} | "
                  f"fit={mean_fit:.4f} | {elapsed:.1f}s")

            if best_acc >= 0.95:
                break

    return best_accuracy, model.num_params_per_member, time.time() - start_time, history


def main():
    print("=" * 70)
    print("Experiment 005: Fitness Function Comparison")
    print("=" * 70)

    fitness_functions = [
        (fitness_mse, "MSE"),
        (fitness_cross_entropy, "CrossEntropy"),
        (fitness_log_softmax, "LogSoftmax"),
        (fitness_soft_accuracy, "SoftAccuracy"),
        (fitness_rank_based, "RankBased"),
    ]

    # Test on 2-digit problem (where MSE currently fails)
    num_digits = 2
    hidden_dim = 32
    population_size = 500
    mutation_std = 0.1
    max_gens = 2000

    print(f"\nConfiguration:")
    print(f"  Digits: {num_digits}")
    print(f"  Hidden: {hidden_dim}")
    print(f"  Population: {population_size}")
    print(f"  Mutation std: {mutation_std}")
    print(f"  Max generations: {max_gens}")

    results = []

    for fitness_fn, name in fitness_functions:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        best_acc, params, elapsed, history = run_experiment(
            fitness_fn, name, num_digits, hidden_dim,
            population_size, mutation_std, max_gens
        )

        status = "PASS" if best_acc >= 0.9 else ("PARTIAL" if best_acc >= 0.5 else "FAIL")
        results.append((name, params, best_acc, elapsed, status))
        print(f"\n{name}: {best_acc:.4f} ({status})")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Fitness':<15} {'Params':>7} {'Accuracy':>10} {'Time':>8} {'Status':>8}")
    print("-" * 60)
    for name, params, acc, elapsed, status in results:
        print(f"{name:<15} {params:>7} {acc:>10.4f} {elapsed:>7.1f}s {status:>8}")


if __name__ == '__main__':
    main()
