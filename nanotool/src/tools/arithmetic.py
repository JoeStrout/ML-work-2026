"""
Arithmetic tool modules.

These are non-differentiable computation modules that perform exact
arithmetic operations. Networks must learn to interface with these
tools through evolution, not backpropagation.
"""

import torch
from typing import Tuple


class AdditionTool:
    """
    Exact integer addition tool.

    This module takes decoded integer operands, performs exact addition,
    and returns the result. It is non-differentiable by design.
    """

    def __init__(self, max_digits: int = 10):
        """
        Args:
            max_digits: Maximum number of digits in result
        """
        self.max_digits = max_digits
        self.max_value = 10 ** max_digits - 1

    def compute(self, a: int, b: int) -> int:
        """
        Perform exact integer addition.

        Args:
            a: First operand (non-negative integer)
            b: Second operand (non-negative integer)

        Returns:
            Sum a + b, clamped to max_value
        """
        result = a + b
        return min(result, self.max_value)

    def compute_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Perform exact addition on a batch of integer pairs.

        Args:
            a: Batch of first operands (1D tensor of ints)
            b: Batch of second operands (1D tensor of ints)

        Returns:
            Batch of sums
        """
        # Move to CPU and convert to Python ints for exact computation
        a_np = a.cpu().numpy().astype(int)
        b_np = b.cpu().numpy().astype(int)

        results = a_np + b_np
        results = results.clip(0, self.max_value)

        return torch.from_numpy(results).to(a.device)


class MultiplicationTool:
    """
    Exact integer multiplication tool.
    """

    def __init__(self, max_digits: int = 10):
        self.max_digits = max_digits
        self.max_value = 10 ** max_digits - 1

    def compute(self, a: int, b: int) -> int:
        """Perform exact integer multiplication."""
        result = a * b
        return min(result, self.max_value)

    def compute_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Perform exact multiplication on a batch."""
        a_np = a.cpu().numpy().astype(int)
        b_np = b.cpu().numpy().astype(int)

        results = a_np * b_np
        results = results.clip(0, self.max_value)

        return torch.from_numpy(results).to(a.device)


class ModularArithmeticTool:
    """
    Exact modular arithmetic tool.
    """

    def __init__(self, modulus: int = 97):
        self.modulus = modulus

    def add(self, a: int, b: int) -> int:
        """Compute (a + b) mod p."""
        return (a + b) % self.modulus

    def multiply(self, a: int, b: int) -> int:
        """Compute (a * b) mod p."""
        return (a * b) % self.modulus

    def add_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batch modular addition."""
        return (a + b) % self.modulus

    def multiply_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batch modular multiplication."""
        return (a * b) % self.modulus
