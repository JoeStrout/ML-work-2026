"""
Number encoding schemes for neural network input/output.

These encodings convert between integers and tensor representations
suitable for neural network processing.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class DigitEncoding:
    """
    Encode integers as sequences of one-hot digit vectors.

    Example: 42 with num_digits=3 → [[0,0,0,0,1,0,0,0,0,0],  # 4
                                      [0,0,1,0,0,0,0,0,0,0],  # 2
                                      [1,0,0,0,0,0,0,0,0,0]]  # 0 (padding)

    Digits are ordered from most significant to least significant.
    """

    def __init__(self, num_digits: int, base: int = 10):
        """
        Args:
            num_digits: Fixed number of digit positions
            base: Number base (default 10 for decimal)
        """
        self.num_digits = num_digits
        self.base = base
        self.encoding_dim = num_digits * base  # Flattened one-hot

    def encode(self, x: int) -> torch.Tensor:
        """
        Encode a single integer.

        Args:
            x: Non-negative integer

        Returns:
            Flattened one-hot tensor of shape (num_digits * base,)
        """
        digits = []
        val = x
        for _ in range(self.num_digits):
            digits.append(val % self.base)
            val //= self.base
        digits = digits[::-1]  # Most significant first

        one_hot = torch.zeros(self.num_digits, self.base)
        for i, d in enumerate(digits):
            one_hot[i, d] = 1.0

        return one_hot.flatten()

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of integers.

        Args:
            x: Tensor of integers, shape (batch_size,)

        Returns:
            Tensor of shape (batch_size, num_digits * base)
        """
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, self.num_digits, self.base)

        for d in range(self.num_digits):
            digit_idx = self.num_digits - 1 - d
            digit_val = (x // (self.base ** d)) % self.base
            result[:, digit_idx, :] = F.one_hot(digit_val.long(), self.base).float()

        return result.view(batch_size, -1)

    def decode(self, encoding: torch.Tensor) -> int:
        """
        Decode one-hot encoding back to integer.

        Args:
            encoding: Flattened one-hot tensor

        Returns:
            Decoded integer
        """
        encoding = encoding.view(self.num_digits, self.base)
        digits = encoding.argmax(dim=1)

        value = 0
        for d in digits:
            value = value * self.base + d.item()
        return value

    def decode_batch(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of encodings.

        Args:
            encoding: Tensor of shape (batch_size, num_digits * base)

        Returns:
            Tensor of integers, shape (batch_size,)
        """
        batch_size = encoding.shape[0]
        encoding = encoding.view(batch_size, self.num_digits, self.base)
        digits = encoding.argmax(dim=2)  # (batch_size, num_digits)

        # Convert digit sequence to integers
        multipliers = torch.tensor(
            [self.base ** (self.num_digits - 1 - i) for i in range(self.num_digits)],
            device=encoding.device
        ).float()

        values = (digits.float() * multipliers).sum(dim=1)
        return values.long()


class BinaryEncoding:
    """
    Encode integers as binary vectors.

    More compact than one-hot, but may be harder for networks to learn.
    """

    def __init__(self, num_bits: int):
        """
        Args:
            num_bits: Number of bits (max value = 2^num_bits - 1)
        """
        self.num_bits = num_bits
        self.encoding_dim = num_bits
        self.max_value = 2 ** num_bits - 1

    def encode(self, x: int) -> torch.Tensor:
        """Encode integer as binary vector."""
        x = min(x, self.max_value)
        bits = [(x >> i) & 1 for i in range(self.num_bits)]
        return torch.tensor(bits, dtype=torch.float32)

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Encode batch of integers as binary vectors."""
        x = x.clamp(0, self.max_value).long()
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, self.num_bits)

        for i in range(self.num_bits):
            result[:, i] = (x >> i) & 1

        return result.float()

    def decode(self, encoding: torch.Tensor) -> int:
        """Decode binary vector to integer."""
        bits = (encoding > 0.5).long()
        value = sum(b.item() << i for i, b in enumerate(bits))
        return value

    def decode_batch(self, encoding: torch.Tensor) -> torch.Tensor:
        """Decode batch of binary vectors."""
        bits = (encoding > 0.5).long()
        multipliers = torch.tensor(
            [2 ** i for i in range(self.num_bits)],
            device=encoding.device
        )
        return (bits * multipliers).sum(dim=1)


class ScalarEncoding:
    """
    Simple normalized scalar encoding.

    Maps integers to [0, 1] range. Simple but loses precision for large numbers.
    """

    def __init__(self, max_value: int):
        self.max_value = max_value
        self.encoding_dim = 1

    def encode(self, x: int) -> torch.Tensor:
        return torch.tensor([x / self.max_value], dtype=torch.float32)

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() / self.max_value).unsqueeze(1)

    def decode(self, encoding: torch.Tensor) -> int:
        return int(round(encoding.item() * self.max_value))

    def decode_batch(self, encoding: torch.Tensor) -> torch.Tensor:
        return (encoding.squeeze(1) * self.max_value).round().long()
