#!/usr/bin/env python3
"""
Debug autoencoder to understand why ES isn't learning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.evolution.es import EvolutionStrategy, flatten_params, unflatten_params
from src.encodings.numbers import DigitEncoding


def test_simple_linear():
    """Can ES learn a simple 2->1 linear mapping?"""
    print("=" * 60)
    print("Test 1: Simple linear y = 2*x1 + 3*x2")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1]

    # Model: y = w1*x1 + w2*x2 + b
    num_params = 3

    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.5,
        learning_rate=0.5,
        population_size=50,
        antithetic=False,
        seed=42
    )

    def compute_loss(params):
        w1, w2, b = params
        pred = w1 * X[:, 0] + w2 * X[:, 1] + b
        return np.mean((pred - y) ** 2)

    print(f"Initial loss: {compute_loss(es.get_params()):.4f}")
    print(f"Initial params: {es.get_params()}")

    for gen in range(200):
        epsilon, population = es.ask()
        fitness = np.array([-compute_loss(p) for p in population])
        es.tell(epsilon, fitness)

        if gen % 50 == 0:
            print(f"Gen {gen}: loss={compute_loss(es.get_params()):.4f}, params={es.get_params()}")

    final_params = es.get_params()
    print(f"\nFinal params: w1={final_params[0]:.4f}, w2={final_params[1]:.4f}, b={final_params[2]:.4f}")
    print(f"Target: w1=2, w2=3, b=0")
    print("PASS" if compute_loss(final_params) < 0.1 else "FAIL")


def test_pytorch_linear():
    """Can ES learn a PyTorch linear layer?"""
    print("\n" + "=" * 60)
    print("Test 2: PyTorch Linear(2, 1)")
    print("=" * 60)

    # Generate data
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn(100, 2)
    true_w = torch.tensor([[2.0, 3.0]])
    y = X @ true_w.T  # (100, 1)

    # Model
    model = nn.Linear(2, 1, bias=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params}")

    params = flatten_params(model)
    print(f"Initial params: {params}")

    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.5,
        learning_rate=0.5,
        population_size=50,
        antithetic=False,
        seed=42
    )
    es.set_params(params)

    def compute_loss():
        with torch.no_grad():
            pred = model(X)
            return F.mse_loss(pred, y).item()

    print(f"Initial loss: {compute_loss():.4f}")

    for gen in range(200):
        epsilon, population = es.ask()
        fitness = np.zeros(len(population))

        for i, p in enumerate(population):
            unflatten_params(model, p)
            fitness[i] = -compute_loss()

        es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

        if gen % 50 == 0:
            print(f"Gen {gen}: loss={compute_loss():.4f}, params={flatten_params(model)}")

    print(f"\nFinal params: {flatten_params(model)}")
    print(f"Target: [2, 3]")
    print("PASS" if compute_loss() < 0.1 else "FAIL")


def test_identity_no_hidden():
    """Can ES learn identity without hidden layer?"""
    print("\n" + "=" * 60)
    print("Test 3: Identity mapping (no hidden layer)")
    print("=" * 60)

    input_dim = 10  # 1 digit one-hot

    model = nn.Linear(input_dim, input_dim)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params}")

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

    def compute_loss():
        with torch.no_grad():
            pred = model(X)
            return F.mse_loss(pred, X).item()

    def compute_accuracy():
        with torch.no_grad():
            pred = model(X)
            pred_digits = pred.argmax(dim=1)
            true_digits = X.argmax(dim=1)
            return (pred_digits == true_digits).float().mean().item()

    print(f"Initial loss: {compute_loss():.4f}, accuracy: {compute_accuracy():.4f}")

    for gen in range(500):
        epsilon, population = es.ask()
        fitness = np.zeros(len(population))

        for i, p in enumerate(population):
            unflatten_params(model, p)
            fitness[i] = -compute_loss()

        es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

        if gen % 100 == 0:
            print(f"Gen {gen}: loss={compute_loss():.4f}, acc={compute_accuracy():.4f}")

    print(f"\nFinal accuracy: {compute_accuracy():.4f}")
    print("PASS" if compute_accuracy() > 0.9 else "FAIL")


def test_identity_with_hidden():
    """Can ES learn identity with hidden layer?"""
    print("\n" + "=" * 60)
    print("Test 4: Identity mapping (WITH hidden layer)")
    print("=" * 60)

    input_dim = 10  # 1 digit one-hot
    hidden_dim = 20

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim)
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params}")

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

    def compute_loss():
        with torch.no_grad():
            pred = model(X)
            return F.mse_loss(pred, X).item()

    def compute_accuracy():
        with torch.no_grad():
            pred = model(X)
            pred_digits = pred.argmax(dim=1)
            true_digits = X.argmax(dim=1)
            return (pred_digits == true_digits).float().mean().item()

    print(f"Initial loss: {compute_loss():.4f}, accuracy: {compute_accuracy():.4f}")

    for gen in range(1000):
        epsilon, population = es.ask()
        fitness = np.zeros(len(population))

        for i, p in enumerate(population):
            unflatten_params(model, p)
            fitness[i] = -compute_loss()

        es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

        if gen % 200 == 0:
            print(f"Gen {gen}: loss={compute_loss():.4f}, acc={compute_accuracy():.4f}")

    print(f"\nFinal accuracy: {compute_accuracy():.4f}")
    print("PASS" if compute_accuracy() > 0.9 else "FAIL")


def debug_output():
    """Debug what the autoencoder is actually outputting."""
    print("\n" + "=" * 60)
    print("Debug: What is the autoencoder outputting?")
    print("=" * 60)

    input_dim = 40  # 4 digits
    hidden_dim = 64

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim)
    )

    # Generate one sample
    X = torch.zeros(1, input_dim)
    X[0, 0] = 1.0   # First digit = 0
    X[0, 10] = 1.0  # Second digit = 0
    X[0, 20] = 1.0  # Third digit = 0
    X[0, 30] = 1.0  # Fourth digit = 0

    print(f"Input (number 0000):")
    print(f"  Shape: {X.shape}")
    print(f"  Non-zero positions: {torch.where(X > 0)[1].tolist()}")

    with torch.no_grad():
        out = model(X)
        print(f"\nOutput (random init):")
        print(f"  Min: {out.min():.4f}, Max: {out.max():.4f}, Mean: {out.mean():.4f}")
        print(f"  Argmax per digit: {out.view(4, 10).argmax(dim=1).tolist()}")

    # What should the loss be for random initialization?
    loss = F.mse_loss(out, X)
    print(f"\nMSE loss: {loss.item():.4f}")


if __name__ == '__main__':
    test_simple_linear()
    test_pytorch_linear()
    test_identity_no_hidden()
    test_identity_with_hidden()
    debug_output()
