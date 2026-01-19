#!/usr/bin/env python3
"""
Experiment 003c: Autoencoder Test (High Learning Rate)

After fixing the gradient normalization bug, try higher learning rates.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time

from src.evolution.es import EvolutionStrategy, flatten_params, unflatten_params
from src.encodings.numbers import DigitEncoding


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def generate_batch(batch_size: int, num_digits: int, device: str = 'cpu'):
    max_val = 10 ** num_digits - 1
    numbers = torch.randint(0, max_val + 1, (batch_size,))
    encoding = DigitEncoding(num_digits)
    x = encoding.encode_batch(numbers).to(device)
    return x, numbers


def compute_accuracy(model, num_digits: int, num_samples: int = 1000, device: str = 'cpu'):
    model.eval()
    encoding = DigitEncoding(num_digits)
    x, numbers = generate_batch(num_samples, num_digits, device)

    with torch.no_grad():
        x_recon = model(x)
        recon_numbers = encoding.decode_batch(x_recon)
        accuracy = (recon_numbers == numbers.to(device)).float().mean().item()

        x_digits = x.view(-1, num_digits, 10).argmax(dim=2)
        x_recon_digits = x_recon.view(-1, num_digits, 10).argmax(dim=2)
        digit_accuracy = (x_digits == x_recon_digits).float().mean().item()

    return accuracy, digit_accuracy


def main():
    print("=" * 60)
    print("Experiment 003c: Autoencoder (High LR)")
    print("=" * 60)

    # Try much higher learning rate
    num_digits = 4
    hidden_dim = 64
    batch_size = 256
    population_size = 100
    sigma = 0.1
    learning_rate = 1.0  # Much higher!
    weight_decay = 0.0
    max_generations = 5000
    eval_frequency = 50
    target_accuracy = 0.95

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Learning rate: {learning_rate}")

    input_dim = num_digits * 10
    model = SimpleAutoencoder(input_dim, hidden_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    params = flatten_params(model)
    es = EvolutionStrategy(
        num_params=len(params),
        sigma=sigma,
        learning_rate=learning_rate,
        population_size=population_size,
        weight_decay=weight_decay,
        seed=42
    )
    es.set_params(params)

    print(f"\nTraining for up to {max_generations} generations")
    print("-" * 60)

    start_time = time.time()

    for gen in range(max_generations):
        x, _ = generate_batch(batch_size, num_digits, device)
        epsilon, population = es.ask()

        fitness = np.zeros(population_size)
        for i in range(population_size):
            unflatten_params(model, population[i])
            with torch.no_grad():
                x_recon = model(x)
                loss = F.mse_loss(x_recon, x).item()
            fitness[i] = -loss

        mean_fitness = es.tell(epsilon, fitness)
        unflatten_params(model, es.get_params())

        if gen % eval_frequency == 0:
            accuracy, digit_accuracy = compute_accuracy(model, num_digits, 2000, device)
            elapsed = time.time() - start_time
            print(f"Gen {gen:5d} | fit={mean_fitness:.6f} | acc={accuracy:.4f} | digit={digit_accuracy:.4f} | {elapsed:.1f}s")

            if accuracy >= target_accuracy:
                print(f"\nTarget reached!")
                break

    final_acc, final_digit = compute_accuracy(model, num_digits, 5000, device)
    print(f"\nFinal: acc={final_acc:.4f}, digit={final_digit:.4f}")
    print("SUCCESS!" if final_acc > 0.9 else "FAILURE")


if __name__ == '__main__':
    main()
