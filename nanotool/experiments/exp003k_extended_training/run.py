#!/usr/bin/env python3
"""
Extended training with SiLU activation and larger population.
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


def test_configurations():
    """Test different configurations."""
    print("=" * 60)
    print("Testing configurations for autoencoder learning")
    print("=" * 60)

    input_dim = 10
    hidden_dim = 20
    max_gens = 3000

    # Generate one-hot data
    X = torch.zeros(100, input_dim)
    for i in range(100):
        X[i, i % input_dim] = 1.0

    configs = [
        # (pop_size, sigma, lr, activation)
        (50, 0.5, 0.5, nn.SiLU()),
        (100, 0.5, 0.5, nn.SiLU()),
        (200, 0.5, 0.5, nn.SiLU()),
        (100, 0.3, 0.3, nn.SiLU()),
        (100, 0.1, 0.5, nn.SiLU()),
        (100, 0.5, 0.5, nn.LeakyReLU(0.1)),
    ]

    for pop_size, sigma, lr, activation in configs:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, input_dim)
        )
        num_params = sum(p.numel() for p in model.parameters())

        params = flatten_params(model)
        es = EvolutionStrategy(
            num_params=num_params,
            sigma=sigma,
            learning_rate=lr,
            population_size=pop_size,
            antithetic=False,
            seed=42
        )
        es.set_params(params)

        def compute_accuracy():
            with torch.no_grad():
                pred = model(X)
                return (pred.argmax(dim=1) == X.argmax(dim=1)).float().mean().item()

        def compute_loss():
            with torch.no_grad():
                return F.mse_loss(model(X), X).item()

        start = time.time()
        for gen in range(max_gens):
            epsilon, population = es.ask()
            fitness = np.zeros(len(population))
            for i, p in enumerate(population):
                unflatten_params(model, p)
                fitness[i] = -compute_loss()
            es.tell(epsilon, fitness)
            unflatten_params(model, es.get_params())

        elapsed = time.time() - start
        acc = compute_accuracy()
        act_name = type(activation).__name__
        print(f"pop={pop_size:3d}, σ={sigma:.1f}, lr={lr:.1f}, {act_name:10s} -> acc={acc:.2f} ({elapsed:.1f}s)")


def test_full_autoencoder():
    """Test full 4-digit autoencoder with best settings."""
    print("\n" + "=" * 60)
    print("Full 4-digit autoencoder with SiLU")
    print("=" * 60)

    num_digits = 4
    input_dim = num_digits * 10
    hidden_dim = 64

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, input_dim)
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params}")

    # Generate one-hot data
    batch_size = 256
    X = torch.zeros(batch_size, input_dim)
    for i in range(batch_size):
        for d in range(num_digits):
            digit = np.random.randint(0, 10)
            X[i, d * 10 + digit] = 1.0

    params = flatten_params(model)
    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.5,
        learning_rate=0.5,
        population_size=200,
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

    print(f"Initial accuracy: {compute_accuracy():.4f}")
    print("-" * 60)

    start = time.time()
    for gen in range(5000):
        epsilon, population = es.ask()
        fitness = np.zeros(len(population))
        for i, p in enumerate(population):
            unflatten_params(model, p)
            fitness[i] = -compute_loss()
        es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

        if gen % 500 == 0:
            acc = compute_accuracy()
            elapsed = time.time() - start
            print(f"Gen {gen:5d} | acc={acc:.4f} | {elapsed:.1f}s")

            if acc >= 0.95:
                print("Target reached!")
                break

    final_acc = compute_accuracy()
    print(f"\nFinal accuracy: {final_acc:.4f}")
    print("SUCCESS!" if final_acc >= 0.95 else ("PARTIAL" if final_acc >= 0.8 else "FAILURE"))


if __name__ == '__main__':
    test_configurations()
    test_full_autoencoder()
