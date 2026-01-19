"""
Neural network architectures for tool-augmented learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..tools.arithmetic import AdditionTool
from ..encodings.numbers import DigitEncoding


class SimpleMLP(nn.Module):
    """
    Simple feedforward MLP baseline (no tool).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu'
    ):
        super().__init__()

        self.activation = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
        }[activation]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class ToolAugmentedMLP(nn.Module):
    """
    MLP with embedded arithmetic tool module.

    Architecture:
    - Encoder: input → hidden representation
    - Two pathways:
      1. Direct: hidden → output (learned approximation)
      2. Tool: hidden → decode to ints → exact add → encode result
    - Gate: blends the two pathways
    """

    def __init__(
        self,
        num_digits: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        self.num_digits = num_digits
        self.encoding = DigitEncoding(num_digits)
        self.tool = AdditionTool(max_digits=num_digits + 1)  # +1 for carry

        # Input: two encoded numbers
        input_dim = 2 * self.encoding.encoding_dim
        # Output: encoded sum (may have one more digit)
        output_dim = DigitEncoding(num_digits + 1).encoding_dim

        self.output_encoding = DigitEncoding(num_digits + 1)

        # Encoder network
        encoder_layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_hidden_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.encoder = nn.ModuleList(encoder_layers)

        # Direct pathway (bypasses tool)
        self.direct_head = nn.Linear(hidden_dim, output_dim)

        # Tool pathway: decoder from hidden to digit logits for each operand
        self.tool_decoder_a = nn.Linear(hidden_dim, self.encoding.encoding_dim)
        self.tool_decoder_b = nn.Linear(hidden_dim, self.encoding.encoding_dim)

        # Tool pathway: encoder from tool output back to hidden
        self.tool_encoder = nn.Linear(output_dim, hidden_dim)

        # Tool pathway: final projection
        self.tool_head = nn.Linear(hidden_dim, output_dim)

        # Gate (learned parameter)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with tool-augmented computation.

        Args:
            x: Input tensor (batch_size, 2 * encoding_dim)
               Concatenated encodings of two operands
            return_gate: If True, also return gate activation

        Returns:
            output: Predicted sum encoding (batch_size, output_encoding_dim)
            gate: (optional) Gate activation value
        """
        batch_size = x.shape[0]

        # Encode through shared encoder
        h = x
        for layer in self.encoder:
            h = F.relu(layer(h))

        # === Direct pathway ===
        direct_out = self.direct_head(h)

        # === Tool pathway ===
        # Decode to discrete integers
        logits_a = self.tool_decoder_a(h)
        logits_b = self.tool_decoder_b(h)

        # Reshape to (batch, num_digits, base) and get digit predictions
        logits_a = logits_a.view(batch_size, self.num_digits, -1)
        logits_b = logits_b.view(batch_size, self.num_digits, -1)

        # Argmax to get discrete digits (non-differentiable)
        digits_a = logits_a.argmax(dim=2)  # (batch, num_digits)
        digits_b = logits_b.argmax(dim=2)

        # Convert digit sequences to integers
        base = 10
        multipliers = torch.tensor(
            [base ** (self.num_digits - 1 - i) for i in range(self.num_digits)],
            device=x.device
        ).float()

        int_a = (digits_a.float() * multipliers).sum(dim=1).long()
        int_b = (digits_b.float() * multipliers).sum(dim=1).long()

        # Exact addition via tool (non-differentiable)
        int_sum = self.tool.compute_batch(int_a, int_b)

        # Encode result back
        sum_encoded = self.output_encoding.encode_batch(int_sum).to(x.device)

        # Process through tool encoder and head
        tool_h = F.relu(self.tool_encoder(sum_encoded))
        tool_out = self.tool_head(tool_h)

        # === Gate and blend ===
        gate = torch.sigmoid(self.gate_logit)
        output = gate * tool_out + (1 - gate) * direct_out

        if return_gate:
            return output, gate
        return output

    def get_gate_value(self) -> float:
        """Get current gate activation."""
        return torch.sigmoid(self.gate_logit).item()


class ToolAugmentedMLPv2(nn.Module):
    """
    Alternative architecture: tool as a side-channel.

    The network can query the tool and receive the answer as additional input,
    but still makes its own final prediction. This may be easier to learn.
    """

    def __init__(
        self,
        num_digits: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        self.num_digits = num_digits
        self.input_encoding = DigitEncoding(num_digits)
        self.output_encoding = DigitEncoding(num_digits + 1)
        self.tool = AdditionTool(max_digits=num_digits + 1)

        input_dim = 2 * self.input_encoding.encoding_dim
        output_dim = self.output_encoding.encoding_dim

        # Query network: decides what to send to tool
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.input_encoding.encoding_dim)
        )

        # Main network: takes original input + tool response
        # Tool response is encoded sum
        main_input_dim = input_dim + output_dim
        layers = [nn.Linear(main_input_dim, hidden_dim)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.main_net = nn.ModuleList(layers)

        # Use-tool gate
        self.use_tool_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Generate tool query
        query = self.query_net(x)

        # Decode query to integers
        query_a = query[:, :self.input_encoding.encoding_dim]
        query_b = query[:, self.input_encoding.encoding_dim:]

        int_a = self.input_encoding.decode_batch(query_a)
        int_b = self.input_encoding.decode_batch(query_b)

        # Get tool response
        int_sum = self.tool.compute_batch(int_a, int_b)
        tool_response = self.output_encoding.encode_batch(int_sum).to(x.device)

        # Gate the tool response
        gate = torch.sigmoid(self.use_tool_logit)
        gated_response = gate * tool_response

        # Concatenate with original input
        main_input = torch.cat([x, gated_response], dim=1)

        # Forward through main network
        h = main_input
        for layer in self.main_net[:-1]:
            h = F.relu(layer(h))
        output = self.main_net[-1](h)

        return output
