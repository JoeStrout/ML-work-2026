#!/usr/bin/env python3
"""
Experiment 003b: Autoencoder Test (Fixed)

Try different loss function and hyperparameters.
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
    """Simple MLP autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


def generate_batch(batch_size: int, num_digits: int, device: str = 'cpu'):
    """Generate batch of one-hot encoded numbers."""
    max_val = 10 ** num_digits - 1
    numbers = torch.randint(0, max_val + 1, (batch_size,))

    encoding = DigitEncoding(num_digits)
    x = encoding.encode_batch(numbers).to(device)

    return x, numbers


def compute_accuracy(model, num_digits: int, num_samples: int = 1000, device: str = 'cpu'):
    """Compute reconstruction accuracy."""
    model.eval()
    encoding = DigitEncoding(num_digits)

    x, numbers = generate_batch(num_samples, num_digits, device)

    with torch.no_grad():
        x_reconstructed = model(x)

        # Decode reconstructions
        reconstructed_numbers = encoding.decode_batch(x_reconstructed)

        # Exact match accuracy
        correct = (reconstructed_numbers == numbers.to(device)).float()
        accuracy = correct.mean().item()

        # Per-digit accuracy
        x_digits = x.view(-1, num_digits, 10).argmax(dim=2)
        x_recon_digits = x_reconstructed.view(-1, num_digits, 10).argmax(dim=2)
        digit_accuracy = (x_digits == x_recon_digits).float().mean().item()

    return accuracy, digit_accuracy


def main():
    print("=" * 60)
    print("Experiment 003b: Autoencoder Test (Cross-Entropy)")
    print("=" * 60)

    # Configuration - more aggressive
    num_digits = 4
    hidden_dim = 64
    batch_size = 256
    population_size = 100
    sigma = 0.05  # Reduced from 0.1
    learning_rate = 0.01  # Increased from 0.001
    weight_decay = 0.0  # Removed weight decay
    max_generations = 5000
    eval_frequency = 50
    target_accuracy = 0.99

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Using cross-entropy loss (per digit)")
    print(f"Learning rate: {learning_rate} (10x higher)")
    print(f"No weight decay")

    input_dim = num_digits * 10
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")

    # Create model
    model = SimpleAutoencoder(input_dim, hidden_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Initialize ES
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
        # Generate batch for this generation
        x, _ = generate_batch(batch_size, num_digits, device)

        # Get perturbations
        epsilon, population = es.ask()

        # Evaluate each member
        fitness = np.zeros(population_size)
        for i in range(population_size):
            unflatten_params(model, population[i])
            model.eval()

            with torch.no_grad():
                x_recon = model(x)

                # Cross-entropy loss per digit (better for classification)
                x_recon_reshaped = x_recon.view(-1, num_digits, 10)
                x_target = x.view(-1, num_digits, 10).argmax(dim=2)

                loss = F.cross_entropy(
                    x_recon_reshaped.permute(0, 2, 1),  # (batch, 10, digits)
                    x_target,  # (batch, digits)
                    reduction='mean'
                ).item()

            fitness[i] = -loss

        # Update ES
        mean_fitness = es.tell(epsilon, fitness)

        # Load best params
        unflatten_params(model, es.get_params())

        # Periodic evaluation
        if gen % eval_frequency == 0 or gen == max_generations - 1:
            accuracy, digit_accuracy = compute_accuracy(model, num_digits, 2000, device)

            elapsed = time.time() - start_time
            print(
                f"Gen {gen:5d} | "
                f"fitness={mean_fitness:.4f} | "
                f"acc={accuracy:.4f} | "
                f"digit_acc={digit_accuracy:.4f} | "
                f"time={elapsed:.1f}s"
            )

            # Early stopping
            if accuracy >= target_accuracy:
                print(f"\nReached target accuracy!")
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_acc, final_digit_acc = compute_accuracy(model, num_digits, 5000, device)
    print(f"Accuracy (exact match): {final_acc:.4f}")
    print(f"Digit accuracy: {final_digit_acc:.4f}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s")

    if final_acc >= target_accuracy:
        print("\nSUCCESS!")
    elif final_digit_acc > 0.9:
        print("\nPARTIAL SUCCESS")
    else:
        print("\nFAILURE")


if __name__ == '__main__':
    main()
