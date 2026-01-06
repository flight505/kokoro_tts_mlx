"""LSTM implementation for MLX (MLX doesn't have built-in LSTM)."""

import mlx.core as mx
import mlx.nn as nn


class LSTMCell(nn.Module):
    """Single LSTM cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weights for input and hidden transformations
        # Gates: input, forget, cell, output (4 * hidden_size)
        self.Wx = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def __call__(
        self, x: mx.array, state: tuple[mx.array, mx.array] | None = None
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Forward pass for single timestep.

        Args:
            x: Input tensor of shape (batch, input_size)
            state: Tuple of (h, c) each of shape (batch, hidden_size)

        Returns:
            h: Hidden state of shape (batch, hidden_size)
            (h, c): New state tuple
        """
        batch_size = x.shape[0]

        if state is None:
            h = mx.zeros((batch_size, self.hidden_size))
            c = mx.zeros((batch_size, self.hidden_size))
        else:
            h, c = state

        # Combined linear transformation
        gates = self.Wx(x) + self.Wh(h)

        # Split into individual gates
        i, f, g, o = mx.split(gates, 4, axis=-1)

        # Apply activations
        i = mx.sigmoid(i)  # Input gate
        f = mx.sigmoid(f)  # Forget gate
        g = mx.tanh(g)  # Cell gate
        o = mx.sigmoid(o)  # Output gate

        # Update cell and hidden state
        c_new = f * c + i * g
        h_new = o * mx.tanh(c_new)

        return h_new, (h_new, c_new)


class LSTM(nn.Module):
    """
    Bidirectional LSTM layer.

    Processes sequences in both forward and backward directions,
    concatenating the outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Forward LSTM cells (one per layer)
        self.forward_cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size * (2 if bidirectional else 1),
                hidden_size,
                bias,
            )
            for i in range(num_layers)
        ]

        # Backward LSTM cells (if bidirectional)
        if bidirectional:
            self.backward_cells = [
                LSTMCell(
                    input_size if i == 0 else hidden_size * 2,
                    hidden_size,
                    bias,
                )
                for i in range(num_layers)
            ]

    def _forward_direction(
        self,
        x: mx.array,
        cell: LSTMCell,
        reverse: bool = False,
    ) -> mx.array:
        """Process sequence in one direction."""
        batch_size, seq_len, _ = x.shape
        state = None
        outputs = []

        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for t in indices:
            out, state = cell(x[:, t, :], state)
            outputs.append(out)

        if reverse:
            outputs = outputs[::-1]

        return mx.stack(outputs, axis=1)

    def __call__(
        self, x: mx.array, state: tuple | None = None
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor (batch, seq_len, input_size) if batch_first else (seq_len, batch, input_size)
            state: Initial state (not used in this implementation, kept for API compatibility)

        Returns:
            output: Output tensor (batch, seq_len, hidden_size * num_directions)
            (h_n, c_n): Final states
        """
        if not self.batch_first:
            x = x.transpose(1, 0, 2)

        current_input = x

        for layer_idx in range(self.num_layers):
            # Forward direction
            forward_out = self._forward_direction(
                current_input, self.forward_cells[layer_idx], reverse=False
            )

            if self.bidirectional:
                # Backward direction
                backward_out = self._forward_direction(
                    current_input, self.backward_cells[layer_idx], reverse=True
                )
                # Concatenate forward and backward
                current_input = mx.concatenate([forward_out, backward_out], axis=-1)
            else:
                current_input = forward_out

        output = current_input

        if not self.batch_first:
            output = output.transpose(1, 0, 2)

        # Return dummy states for API compatibility
        h_n = mx.zeros((self.num_layers * (2 if self.bidirectional else 1), x.shape[0], self.hidden_size))
        c_n = mx.zeros_like(h_n)

        return output, (h_n, c_n)
