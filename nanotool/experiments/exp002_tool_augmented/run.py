#!/usr/bin/env python3
"""
Experiment 002: Tool-Augmented Addition

Train a tool-augmented MLP to perform multi-digit addition using Evolution Strategies.
The network has access to an embedded exact addition module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import torch
import numpy as np
from pathlib import Path

from src.networks.mlp import ToolAugmentedMLP
from src.training.trainer import ESTrainer, TrainingConfig


def main():
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("Experiment 002: Tool-Augmented Addition")
    print("=" * 60)

    # Extract settings
    num_digits = cfg['task']['num_digits']
    hidden_dim = cfg['network']['hidden_dim']
    num_hidden_layers = cfg['network']['num_hidden_layers']

    print(f"\nTask: {num_digits}-digit addition")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Hidden layers: {num_hidden_layers}")

    # Create model
    model = ToolAugmentedMLP(
        num_digits=num_digits,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    print(f"Initial gate value: {model.get_gate_value():.3f}")

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
    print(f"Final gate value: {final_metrics['gate']:.4f}")

    # Analyze gate evolution
    gate_history = [g for g in history['gate_value'] if g is not None]
    if gate_history:
        print(f"\nGate evolution:")
        print(f"  Start: {gate_history[0]:.4f}")
        print(f"  End:   {gate_history[-1]:.4f}")
        print(f"  Min:   {min(gate_history):.4f}")
        print(f"  Max:   {max(gate_history):.4f}")

    # Save final results summary
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    summary = {
        'experiment': 'exp002_tool_augmented',
        'config': cfg,
        'final_accuracy': final_metrics['accuracy'],
        'final_digit_accuracy': final_metrics['digit_accuracy'],
        'final_gate': final_metrics['gate'],
        'num_params': num_params,
        'gate_evolution': {
            'start': gate_history[0] if gate_history else None,
            'end': gate_history[-1] if gate_history else None,
        }
    }

    with open(results_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
