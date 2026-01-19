#!/usr/bin/env python3
"""
Experiment 003d: ES Diagnostic

Minimal test to verify ES can optimize a simple function.
If ES can't minimize f(x) = ||x - target||^2, then the core algorithm is broken.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from src.evolution.es import EvolutionStrategy


def test_quadratic():
    """Can ES minimize f(x) = ||x - target||^2?"""
    print("=" * 60)
    print("Test 1: Quadratic minimization")
    print("=" * 60)

    num_params = 100
    target = np.random.randn(num_params) * 10  # Random target in [-10, 10]

    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.1,
        learning_rate=1.0,
        population_size=50,
        weight_decay=0.0,
        seed=42
    )

    # Start from zeros
    es.set_params(np.zeros(num_params))

    print(f"Target norm: {np.linalg.norm(target):.4f}")
    print(f"Initial distance: {np.linalg.norm(es.get_params() - target):.4f}")
    print("-" * 60)

    for gen in range(500):
        epsilon, population = es.ask()

        # Fitness = negative squared distance to target
        fitness = np.array([-np.sum((p - target)**2) for p in population])

        mean_fitness = es.tell(epsilon, fitness)

        if gen % 50 == 0:
            current = es.get_params()
            dist = np.linalg.norm(current - target)
            print(f"Gen {gen:4d} | fitness={mean_fitness:.4f} | dist={dist:.4f}")

    final_dist = np.linalg.norm(es.get_params() - target)
    print(f"\nFinal distance: {final_dist:.4f}")
    print("PASS" if final_dist < 1.0 else "FAIL")
    return final_dist < 1.0


def test_linear():
    """Can ES learn a linear function y = Wx + b?"""
    print("\n" + "=" * 60)
    print("Test 2: Linear regression")
    print("=" * 60)

    # True weights
    np.random.seed(123)
    input_dim = 10
    output_dim = 5
    true_W = np.random.randn(output_dim, input_dim)
    true_b = np.random.randn(output_dim)

    # Generate data
    X = np.random.randn(100, input_dim)
    Y = X @ true_W.T + true_b

    # Flatten true params
    true_params = np.concatenate([true_W.flatten(), true_b])
    num_params = len(true_params)

    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.1,
        learning_rate=1.0,
        population_size=50,
        weight_decay=0.0,
        seed=42
    )

    def compute_loss(params):
        W = params[:input_dim * output_dim].reshape(output_dim, input_dim)
        b = params[input_dim * output_dim:]
        pred = X @ W.T + b
        return np.mean((pred - Y) ** 2)

    print(f"Initial loss: {compute_loss(es.get_params()):.4f}")
    print("-" * 60)

    for gen in range(500):
        epsilon, population = es.ask()
        fitness = np.array([-compute_loss(p) for p in population])
        mean_fitness = es.tell(epsilon, fitness)

        if gen % 50 == 0:
            loss = compute_loss(es.get_params())
            print(f"Gen {gen:4d} | fitness={mean_fitness:.4f} | loss={loss:.6f}")

    final_loss = compute_loss(es.get_params())
    print(f"\nFinal loss: {final_loss:.6f}")
    print("PASS" if final_loss < 0.01 else "FAIL")
    return final_loss < 0.01


def test_autoencoder_numpy():
    """Can ES learn a simple numpy autoencoder (no PyTorch)?"""
    print("\n" + "=" * 60)
    print("Test 3: NumPy Autoencoder")
    print("=" * 60)

    np.random.seed(42)
    input_dim = 40  # 4 digits * 10 one-hot
    hidden_dim = 64

    # Generate one-hot data
    num_samples = 256
    X = np.zeros((num_samples, input_dim))
    for i in range(num_samples):
        for d in range(4):  # 4 digits
            digit = np.random.randint(0, 10)
            X[i, d * 10 + digit] = 1.0

    # Parameters: encoder W1, b1, decoder W2, b2
    num_params = input_dim * hidden_dim + hidden_dim + hidden_dim * input_dim + input_dim
    print(f"Parameters: {num_params}")

    def forward(params, x):
        idx = 0
        W1 = params[idx:idx + input_dim * hidden_dim].reshape(hidden_dim, input_dim)
        idx += input_dim * hidden_dim
        b1 = params[idx:idx + hidden_dim]
        idx += hidden_dim
        W2 = params[idx:idx + hidden_dim * input_dim].reshape(input_dim, hidden_dim)
        idx += hidden_dim * input_dim
        b2 = params[idx:idx + input_dim]

        h = np.maximum(0, x @ W1.T + b1)  # ReLU
        out = h @ W2.T + b2
        return out

    def compute_loss(params):
        pred = forward(params, X)
        return np.mean((pred - X) ** 2)

    def compute_accuracy(params):
        pred = forward(params, X)
        pred_reshaped = pred.reshape(-1, 4, 10)
        X_reshaped = X.reshape(-1, 4, 10)
        pred_digits = np.argmax(pred_reshaped, axis=2)
        true_digits = np.argmax(X_reshaped, axis=2)
        return np.mean(pred_digits == true_digits)

    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.1,
        learning_rate=1.0,
        population_size=100,
        weight_decay=0.0,
        seed=42
    )

    # Initialize with small random weights (like PyTorch default)
    init_params = np.random.randn(num_params) * 0.1
    es.set_params(init_params)

    print(f"Initial loss: {compute_loss(es.get_params()):.4f}")
    print(f"Initial accuracy: {compute_accuracy(es.get_params()):.4f}")
    print("-" * 60)

    for gen in range(1000):
        epsilon, population = es.ask()
        fitness = np.array([-compute_loss(p) for p in population])
        mean_fitness = es.tell(epsilon, fitness)

        if gen % 100 == 0:
            loss = compute_loss(es.get_params())
            acc = compute_accuracy(es.get_params())
            print(f"Gen {gen:4d} | fitness={mean_fitness:.6f} | loss={loss:.6f} | acc={acc:.4f}")

    final_loss = compute_loss(es.get_params())
    final_acc = compute_accuracy(es.get_params())
    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"Final accuracy: {final_acc:.4f}")
    print("PASS" if final_acc > 0.9 else "FAIL")
    return final_acc > 0.9


if __name__ == '__main__':
    results = []
    results.append(("Quadratic", test_quadratic()))
    results.append(("Linear", test_linear()))
    results.append(("Autoencoder", test_autoencoder_numpy()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
