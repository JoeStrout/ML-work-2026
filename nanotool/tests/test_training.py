#!/usr/bin/env python3
"""
Quick test of the training loop.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.networks.mlp import SimpleMLP, ToolAugmentedMLP
from src.encodings.numbers import DigitEncoding
from src.training.trainer import ESTrainer, TrainingConfig


def test_baseline_training():
    """Test baseline MLP training for a few generations."""
    print("Testing baseline MLP training...")

    num_digits = 2  # Simple 2-digit addition
    input_enc = DigitEncoding(num_digits)
    output_enc = DigitEncoding(num_digits + 1)

    model = SimpleMLP(
        input_dim=2 * input_enc.encoding_dim,
        hidden_dims=[64, 64],  # Larger network
        output_dim=output_enc.encoding_dim
    )

    config = TrainingConfig(
        num_digits=num_digits,
        batch_size=256,
        population_size=50,
        sigma=0.1,  # Larger sigma for more exploration
        learning_rate=0.001,  # Smaller learning rate
        weight_decay=0.01,  # Regularization
        generations=100,
        eval_frequency=20,
        seed=42,
        device='cpu'
    )

    trainer = ESTrainer(model, config)
    history = trainer.train(verbose=True)

    print(f"Final accuracy: {history['train_accuracy'][-1]:.4f}")
    print("Baseline training test: PASSED\n")


def test_tool_augmented_training():
    """Test tool-augmented MLP training for a few generations."""
    print("Testing tool-augmented MLP training...")

    num_digits = 2

    model = ToolAugmentedMLP(
        num_digits=num_digits,
        hidden_dim=64,
        num_hidden_layers=2
    )

    config = TrainingConfig(
        num_digits=num_digits,
        batch_size=256,
        population_size=50,
        sigma=0.1,
        learning_rate=0.001,
        weight_decay=0.01,
        generations=100,
        eval_frequency=20,
        seed=42,
        device='cpu'
    )

    trainer = ESTrainer(model, config)
    history = trainer.train(verbose=True)

    print(f"Final accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"Final gate: {history['gate_value'][-1]:.4f}")
    print("Tool-augmented training test: PASSED\n")


if __name__ == '__main__':
    test_baseline_training()
    print("-" * 60)
    test_tool_augmented_training()
    print("=" * 60)
    print("All training tests passed!")
