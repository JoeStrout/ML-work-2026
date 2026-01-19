"""
Basic tests for core components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.encodings.numbers import DigitEncoding, BinaryEncoding
from src.tools.arithmetic import AdditionTool
from src.networks.mlp import SimpleMLP, ToolAugmentedMLP
from src.evolution.es import EvolutionStrategy, flatten_params, unflatten_params


def test_digit_encoding():
    """Test digit encoding roundtrip."""
    enc = DigitEncoding(num_digits=4)

    # Single value
    for val in [0, 1, 42, 999, 1234, 9999]:
        encoded = enc.encode(val)
        decoded = enc.decode(encoded)
        assert decoded == val, f"Failed for {val}: got {decoded}"

    # Batch
    batch = torch.tensor([0, 42, 999, 1234])
    encoded = enc.encode_batch(batch)
    decoded = enc.decode_batch(encoded)
    assert torch.equal(decoded, batch), f"Batch failed: {decoded} vs {batch}"

    print("Digit encoding: PASSED")


def test_binary_encoding():
    """Test binary encoding roundtrip."""
    enc = BinaryEncoding(num_bits=16)

    for val in [0, 1, 42, 255, 1000, 65535]:
        encoded = enc.encode(val)
        decoded = enc.decode(encoded)
        assert decoded == val, f"Failed for {val}: got {decoded}"

    print("Binary encoding: PASSED")


def test_addition_tool():
    """Test addition tool."""
    tool = AdditionTool(max_digits=5)

    # Single computations
    assert tool.compute(123, 456) == 579
    assert tool.compute(99999, 1) == 99999  # Clamped

    # Batch
    a = torch.tensor([10, 100, 999])
    b = torch.tensor([5, 200, 1])
    result = tool.compute_batch(a, b)
    expected = torch.tensor([15, 300, 1000])
    assert torch.equal(result, expected), f"Got {result}"

    print("Addition tool: PASSED")


def test_simple_mlp():
    """Test simple MLP forward pass."""
    model = SimpleMLP(input_dim=20, hidden_dims=[32, 32], output_dim=10)

    x = torch.randn(8, 20)
    y = model(x)

    assert y.shape == (8, 10), f"Wrong shape: {y.shape}"
    print("Simple MLP: PASSED")


def test_tool_augmented_mlp():
    """Test tool-augmented MLP forward pass."""
    num_digits = 3
    model = ToolAugmentedMLP(num_digits=num_digits, hidden_dim=32)

    # Create input: two 3-digit numbers encoded
    enc = DigitEncoding(num_digits)
    a = torch.tensor([123, 456])
    b = torch.tensor([111, 222])
    x = torch.cat([enc.encode_batch(a), enc.encode_batch(b)], dim=1)

    output, gate = model(x, return_gate=True)

    # Output should be encoding for 4-digit number
    expected_output_dim = DigitEncoding(num_digits + 1).encoding_dim
    assert output.shape == (2, expected_output_dim), f"Wrong shape: {output.shape}"

    print(f"Tool-augmented MLP: PASSED (gate={gate:.3f})")


def test_evolution_strategy():
    """Test ES optimizer."""
    es = EvolutionStrategy(num_params=100, population_size=20, seed=42)

    # Run a few iterations on a simple quadratic
    target = np.random.randn(100)

    for gen in range(10):
        epsilon, population = es.ask()
        # Fitness: negative distance to target
        fitness = -np.sum((population - target) ** 2, axis=1)
        mean_fit = es.tell(epsilon, fitness)

    # Should have improved
    final_dist = np.sum((es.get_params() - target) ** 2)
    assert final_dist < 100 * 100, "ES didn't improve"

    print(f"Evolution strategy: PASSED (final_dist={final_dist:.2f})")


def test_param_flatten_unflatten():
    """Test parameter flattening roundtrip."""
    model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=5)

    original_params = flatten_params(model)
    x = torch.randn(4, 10)
    y1 = model(x).clone()

    # Perturb and restore
    perturbed = original_params + 0.1
    unflatten_params(model, perturbed)
    y2 = model(x).clone()

    unflatten_params(model, original_params)
    y3 = model(x)

    assert not torch.allclose(y1, y2), "Perturbation had no effect"
    assert torch.allclose(y1, y3), "Restore failed"

    print("Param flatten/unflatten: PASSED")


if __name__ == '__main__':
    test_digit_encoding()
    test_binary_encoding()
    test_addition_tool()
    test_simple_mlp()
    test_tool_augmented_mlp()
    test_evolution_strategy()
    test_param_flatten_unflatten()
    print("\n All tests passed!")
