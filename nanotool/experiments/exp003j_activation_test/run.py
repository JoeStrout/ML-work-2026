#!/usr/bin/env python3
"""
Test different activation functions for the autoencoder.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.evolution.es import EvolutionStrategy, flatten_params, unflatten_params


def test_autoencoder(activation_name, activation_fn, input_dim=10, hidden_dim=20, max_gens=1000):
    """Test autoencoder with different activations."""
    print(f"\n{activation_name}:")

    if activation_fn is None:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
    else:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, input_dim)
        )

    num_params = sum(p.numel() for p in model.parameters())

    # Generate one-hot data
    X = torch.zeros(100, input_dim)
    for i in range(100):
        X[i, i % input_dim] = 1.0

    params = flatten_params(model)
    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.5,
        learning_rate=0.5,
        population_size=50,
        antithetic=False,
        seed=42
    )
    es.set_params(params)

    def compute_accuracy():
        with torch.no_grad():
            pred = model(X)
            pred_digits = pred.argmax(dim=1)
            true_digits = X.argmax(dim=1)
            return (pred_digits == true_digits).float().mean().item()

    def compute_loss():
        with torch.no_grad():
            return F.mse_loss(model(X), X).item()

    for gen in range(max_gens):
        epsilon, population = es.ask()
        fitness = np.zeros(len(population))

        for i, p in enumerate(population):
            unflatten_params(model, p)
            fitness[i] = -compute_loss()

        es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

    acc = compute_accuracy()
    loss = compute_loss()
    print(f"  Params: {num_params}, Final acc: {acc:.4f}, loss: {loss:.4f}")
    return acc > 0.9


def main():
    print("=" * 60)
    print("Testing different activation functions")
    print("=" * 60)

    results = []

    # No activation (linear)
    results.append(("Linear", test_autoencoder("Linear", None)))

    # ReLU
    results.append(("ReLU", test_autoencoder("ReLU", nn.ReLU())))

    # LeakyReLU
    results.append(("LeakyReLU", test_autoencoder("LeakyReLU", nn.LeakyReLU(0.1))))

    # Tanh
    results.append(("Tanh", test_autoencoder("Tanh", nn.Tanh())))

    # Sigmoid
    results.append(("Sigmoid", test_autoencoder("Sigmoid", nn.Sigmoid())))

    # GELU
    results.append(("GELU", test_autoencoder("GELU", nn.GELU())))

    # SiLU (Swish)
    results.append(("SiLU", test_autoencoder("SiLU", nn.SiLU())))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")


if __name__ == '__main__':
    main()
