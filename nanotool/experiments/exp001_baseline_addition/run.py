#!/usr/bin/env python3
"""
Experiment 001: Baseline Addition (No Tool)

Train a simple MLP to perform multi-digit addition using Evolution Strategies.
This establishes the baseline performance without tool augmentation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import torch
import numpy as np
from pathlib import Path

from src.networks.mlp import SimpleMLP
from src.encodings.numbers import DigitEncoding
from src.training.trainer import ESTrainer, TrainingConfig


def main():
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("Experiment 001: Baseline Addition (No Tool)")
    print("=" * 60)

    # Extract settings
    num_digits = cfg['task']['num_digits']
    hidden_dims = cfg['network']['hidden_dims']

    # Compute dimensions
    input_encoding = DigitEncoding(num_digits)
    output_encoding = DigitEncoding(num_digits + 1)  # +1 for potential carry

    input_dim = 2 * input_encoding.encoding_dim  # Two operands
    output_dim = output_encoding.encoding_dim

    print(f"\nTask: {num_digits}-digit addition")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")
    print(f"Hidden layers: {hidden_dims}")

    # Create model
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=cfg['network']['activation']
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Set up training config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_config = TrainingConfig(
        num_digits=num_digits,
        batch_size=cfg['training']['batch_size'],
        population_size=cfg['evolution']['population_size'],
        sigma=cfg['evolution']['sigma'],
        learning_rate=cfg['evolution']['learning_rate'],
        weight_decay=cfg['evolution'].get('weight_decay', 0.01),
        generations=cfg['evolution']['generations'],
        eval_frequency=cfg['training']['eval_frequency'],
        seed=cfg['training']['seed'],
        device=device,
        save_dir=str(Path(__file__).parent / 'results'),
        checkpoint_frequency=cfg['logging']['checkpoint_frequency']
    )

    # Train
    trainer = ESTrainer(model, train_config)
    print("\n")
    history = trainer.train(verbose=True)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_metrics = trainer.evaluate(num_samples=2000)
    print(f"Accuracy (exact match): {final_metrics['accuracy']:.4f}")
    print(f"Digit accuracy: {final_metrics['digit_accuracy']:.4f}")

    # Save final results summary
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    summary = {
        'experiment': 'exp001_baseline_addition',
        'config': cfg,
        'final_accuracy': final_metrics['accuracy'],
        'final_digit_accuracy': final_metrics['digit_accuracy'],
        'num_params': num_params,
    }

    with open(results_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
