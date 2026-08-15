"""Fixed-capacity replay storage backed by NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled batch of replay transitions."""

    states: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_states: NDArray[np.float32]
    terminals: NDArray[np.float32]


class ReplayBuffer:
    """Store transitions in a deterministic circular replay buffer."""

    def __init__(self, capacity: int, state_size: int, seed: int) -> None:
        """Allocate replay arrays and initialise the sampling generator."""
        if capacity < 1 or state_size < 1:
            raise ValueError("capacity and state_size must be positive")
        self.capacity = capacity
        self.state_size = state_size
        self._random = np.random.default_rng(seed)
        self._states = np.zeros((capacity, state_size), dtype=np.float32)
        self._actions = np.zeros((capacity, 1), dtype=np.int64)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self._terminals = np.zeros((capacity, 1), dtype=np.float32)
        self._position = 0
        self._size = 0

    def add(
        self,
        state: NDArray[np.floating],
        action: int,
        reward: float,
        next_state: NDArray[np.floating],
        terminal: bool,
    ) -> None:
        """Append one transition, overwriting the oldest item when full."""
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        expected_shape = (self.state_size,)
        if state_array.shape != expected_shape or next_state_array.shape != expected_shape:
            raise ValueError(f"states must have shape {expected_shape}")
        index = self._position
        self._states[index] = state_array
        self._actions[index, 0] = action
        self._rewards[index, 0] = reward
        self._next_states[index] = next_state_array
        self._terminals[index, 0] = float(terminal)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample distinct transition rows with deterministic generator state."""
        if batch_size < 1 or batch_size > self._size:
            raise ValueError("batch_size must be positive and no larger than stored transitions")
        indices = self._random.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminals=self._terminals[indices],
        )

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size
