#!/usr/bin/env python3
"""
Test ES at different scales to find the working range.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from src.evolution.es import EvolutionStrategy, flatten_params, unflatten_params


def test_autoencoder(num_digits, hidden_dim, pop_size=200, max_gens=5000, verbose=True):
    """Test autoencoder with specified configuration."""
    input_dim = num_digits * 10

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, input_dim)
    )
    num_params = sum(p.numel() for p in model.parameters())

    # Generate one-hot data
    batch_size = 256
    np.random.seed(42)
    X = torch.zeros(batch_size, input_dim)
    for i in range(batch_size):
        for d in range(num_digits):
            digit = np.random.randint(0, 10)
            X[i, d * 10 + digit] = 1.0

    params = flatten_params(model)
    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.3,
        learning_rate=0.3,
        population_size=pop_size,
        antithetic=False,
        seed=42
    )
    es.set_params(params)

    def compute_accuracy():
        with torch.no_grad():
            pred = model(X)
            pred_reshaped = pred.view(-1, num_digits, 10)
            X_reshaped = X.view(-1, num_digits, 10)
            return (pred_reshaped.argmax(dim=2) == X_reshaped.argmax(dim=2)).float().mean().item()

    def compute_loss():
        with torch.no_grad():
            return F.mse_loss(model(X), X).item()

    start = time.time()
    best_acc = 0

    for gen in range(max_gens):
        epsilon, population = es.ask()
        fitness = np.zeros(len(population))
        for i, p in enumerate(population):
            unflatten_params(model, p)
            fitness[i] = -compute_loss()
        es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

        if gen % 500 == 0 and verbose:
            acc = compute_accuracy()
            best_acc = max(best_acc, acc)
            elapsed = time.time() - start
            print(f"  Gen {gen:5d} | acc={acc:.4f} | best={best_acc:.4f} | {elapsed:.1f}s")

            if acc >= 0.95:
                break

    final_acc = compute_accuracy()
    total_time = time.time() - start
    return final_acc, num_params, total_time


def main():
    print("=" * 60)
    print("Scale test: Finding working parameter range for ES")
    print("=" * 60)

    results = []

    # Test different configurations
    configs = [
        # (num_digits, hidden_dim, pop_size)
        (1, 10, 100),   # ~120 params
        (1, 20, 100),   # ~230 params
        (2, 10, 100),   # ~320 params
        (2, 20, 100),   # ~640 params
        (2, 32, 200),   # ~1000 params
        (3, 16, 200),   # ~1000 params
        (3, 32, 200),   # ~2000 params
        (4, 16, 200),   # ~1400 params
        (4, 32, 200),   # ~2800 params
        (4, 64, 300),   # ~5500 params
    ]

    print(f"\n{'Digits':>6} {'Hidden':>8} {'Pop':>6} {'Params':>8} {'Acc':>8} {'Time':>8}")
    print("-" * 60)

    for num_digits, hidden_dim, pop_size in configs:
        print(f"\nTesting {num_digits} digits, hidden={hidden_dim}, pop={pop_size}...")
        acc, params, elapsed = test_autoencoder(num_digits, hidden_dim, pop_size, max_gens=3000)
        results.append((num_digits, hidden_dim, pop_size, params, acc, elapsed))
        print(f"{num_digits:>6} {hidden_dim:>8} {pop_size:>6} {params:>8} {acc:>8.4f} {elapsed:>7.1f}s")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Digits':>6} {'Hidden':>8} {'Pop':>6} {'Params':>8} {'Acc':>8} {'Status':>8}")
    print("-" * 60)

    for num_digits, hidden_dim, pop_size, params, acc, elapsed in results:
        status = "PASS" if acc >= 0.95 else ("PARTIAL" if acc >= 0.5 else "FAIL")
        print(f"{num_digits:>6} {hidden_dim:>8} {pop_size:>6} {params:>8} {acc:>8.4f} {status:>8}")


if __name__ == '__main__':
    main()
