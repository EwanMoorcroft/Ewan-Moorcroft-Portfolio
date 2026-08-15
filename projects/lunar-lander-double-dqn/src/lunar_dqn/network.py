"""Neural network used by the value-learning agent."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map a state vector to one action value per discrete action."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        """Build a compact two-layer multilayer perceptron."""
        super().__init__()
        if state_size < 1 or action_size < 1 or hidden_size < 1:
            raise ValueError("network dimensions must be positive")
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return action values for a batch of states."""
        return self.layers(states)
