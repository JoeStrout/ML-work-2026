#!/usr/bin/env python3
"""Quick scale test with fewer configs and generations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys

from src.evolution.es import EvolutionStrategy, flatten_params, unflatten_params

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def test_autoencoder(num_digits, hidden_dim, pop_size=100, max_gens=1500):
    input_dim = num_digits * 10

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, input_dim)
    )
    num_params = sum(p.numel() for p in model.parameters())

    batch_size = 256
    np.random.seed(42)
    X = torch.zeros(batch_size, input_dim)
    for i in range(batch_size):
        for d in range(num_digits):
            X[i, d * 10 + np.random.randint(0, 10)] = 1.0

    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.3,
        learning_rate=0.3,
        population_size=pop_size,
        antithetic=False,
        seed=42
    )
    es.set_params(flatten_params(model))

    def compute_accuracy():
        with torch.no_grad():
            pred = model(X).view(-1, num_digits, 10)
            X_reshaped = X.view(-1, num_digits, 10)
            return (pred.argmax(dim=2) == X_reshaped.argmax(dim=2)).float().mean().item()

    def compute_loss():
        with torch.no_grad():
            return F.mse_loss(model(X), X).item()

    start = time.time()
    for gen in range(max_gens):
        epsilon, population = es.ask()
        fitness = np.array([-F.mse_loss(model(X), X).item()
                           for p in population if unflatten_params(model, p) is None])
        es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

        if gen % 500 == 0:
            print(f"  {num_digits}d h{hidden_dim} | Gen {gen} | acc={compute_accuracy():.3f}", flush=True)

    return compute_accuracy(), num_params, time.time() - start


print("=" * 60, flush=True)
print("Quick Scale Test", flush=True)
print("=" * 60, flush=True)

configs = [
    (1, 10, 50),    # ~120 params
    (2, 16, 100),   # ~550 params
    (2, 32, 100),   # ~1100 params
    (3, 16, 100),   # ~800 params
    (4, 16, 150),   # ~1050 params
    (4, 32, 150),   # ~2100 params
]

print(f"\n{'Config':>12} {'Params':>8} {'Accuracy':>10} {'Time':>8}", flush=True)
print("-" * 45, flush=True)

for num_digits, hidden_dim, pop_size in configs:
    acc, params, elapsed = test_autoencoder(num_digits, hidden_dim, pop_size)
    status = "PASS" if acc > 0.9 else "FAIL"
    print(f"{num_digits}d h{hidden_dim:>2} p{pop_size:>3} {params:>8} {acc:>10.4f} {elapsed:>7.1f}s  {status}", flush=True)
