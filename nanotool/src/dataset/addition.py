"""
Data generation for addition tasks.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from ..encodings.numbers import DigitEncoding


class AdditionDataset:
    """
    Dataset for multi-digit addition problems.

    Generates pairs of N-digit integers and their sums.
    """

    def __init__(
        self,
        num_digits: int,
        num_samples: int,
        seed: Optional[int] = None
    ):
        """
        Args:
            num_digits: Number of digits per operand
            num_samples: Total samples to generate
            seed: Random seed for reproducibility
        """
        self.num_digits = num_digits
        self.num_samples = num_samples
        self.max_value = 10 ** num_digits - 1

        if seed is not None:
            np.random.seed(seed)

        # Generate all data upfront
        self.a = np.random.randint(0, self.max_value + 1, size=num_samples)
        self.b = np.random.randint(0, self.max_value + 1, size=num_samples)
        self.sums = self.a + self.b

        # Set up encodings
        self.input_encoding = DigitEncoding(num_digits)
        self.output_encoding = DigitEncoding(num_digits + 1)  # +1 for carry

    def get_batch(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a random batch of addition problems.

        Returns:
            x: Encoded inputs (batch_size, 2 * input_encoding_dim)
            y: Encoded targets (batch_size, output_encoding_dim)
            targets: Raw integer targets (batch_size,)
        """
        indices = np.random.choice(self.num_samples, size=batch_size, replace=False)

        a = torch.tensor(self.a[indices], dtype=torch.long)
        b = torch.tensor(self.b[indices], dtype=torch.long)
        sums = torch.tensor(self.sums[indices], dtype=torch.long)

        # Encode inputs
        a_enc = self.input_encoding.encode_batch(a)
        b_enc = self.input_encoding.encode_batch(b)
        x = torch.cat([a_enc, b_enc], dim=1).to(device)

        # Encode targets
        y = self.output_encoding.encode_batch(sums).to(device)

        return x, y, sums.to(device)

    def get_all(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all data as tensors."""
        a = torch.tensor(self.a, dtype=torch.long)
        b = torch.tensor(self.b, dtype=torch.long)
        sums = torch.tensor(self.sums, dtype=torch.long)

        a_enc = self.input_encoding.encode_batch(a)
        b_enc = self.input_encoding.encode_batch(b)
        x = torch.cat([a_enc, b_enc], dim=1).to(device)
        y = self.output_encoding.encode_batch(sums).to(device)

        return x, y, sums.to(device)


class ExtrapolationDataset:
    """
    Dataset for testing extrapolation to longer numbers.

    Generates problems with more digits than training.
    """

    def __init__(
        self,
        num_digits: int,
        num_samples: int,
        seed: Optional[int] = None
    ):
        self.num_digits = num_digits
        self.num_samples = num_samples
        self.max_value = 10 ** num_digits - 1

        if seed is not None:
            np.random.seed(seed)

        # Generate data
        self.a = np.random.randint(0, self.max_value + 1, size=num_samples)
        self.b = np.random.randint(0, self.max_value + 1, size=num_samples)
        self.sums = self.a + self.b

        self.input_encoding = DigitEncoding(num_digits)
        self.output_encoding = DigitEncoding(num_digits + 1)

    def get_all(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all extrapolation test data."""
        a = torch.tensor(self.a, dtype=torch.long)
        b = torch.tensor(self.b, dtype=torch.long)
        sums = torch.tensor(self.sums, dtype=torch.long)

        a_enc = self.input_encoding.encode_batch(a)
        b_enc = self.input_encoding.encode_batch(b)
        x = torch.cat([a_enc, b_enc], dim=1).to(device)
        y = self.output_encoding.encode_batch(sums).to(device)

        return x, y, sums.to(device)


def generate_addition_batch(
    batch_size: int,
    num_digits: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a single batch of addition problems on the fly.

    Useful for ES training where we want fresh data each evaluation.
    """
    max_value = 10 ** num_digits - 1

    a = torch.randint(0, max_value + 1, (batch_size,))
    b = torch.randint(0, max_value + 1, (batch_size,))
    sums = a + b

    input_enc = DigitEncoding(num_digits)
    output_enc = DigitEncoding(num_digits + 1)

    a_enc = input_enc.encode_batch(a)
    b_enc = input_enc.encode_batch(b)
    x = torch.cat([a_enc, b_enc], dim=1).to(device)
    y = output_enc.encode_batch(sums).to(device)

    return x, y, sums.to(device)
