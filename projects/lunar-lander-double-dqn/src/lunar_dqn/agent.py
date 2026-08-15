"""Double DQN agent with replay and a softly updated target network."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from lunar_dqn.config import DQNConfig
from lunar_dqn.network import QNetwork
from lunar_dqn.replay import ReplayBuffer


def select_device() -> torch.device:
    """Use Apple Metal acceleration when available and otherwise use the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable local runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class DoubleDQNAgent:
    """Learn discrete action values using Double DQN targets."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: DQNConfig,
        device: torch.device | None = None,
    ) -> None:
        """Create online and target networks plus replay and optimiser state."""
        config.validate()
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = device or select_device()
        self.online = QNetwork(state_size, action_size, config.hidden_size).to(self.device)
        self.target = QNetwork(state_size, action_size, config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, state_size, config.seed)
        self.updates = 0

    def epsilon(self, episode_index: int) -> float:
        """Return linearly decayed exploration probability for an episode index."""
        progress = min(max(episode_index, 0) / self.config.epsilon_decay_episodes, 1.0)
        span = self.config.epsilon_end - self.config.epsilon_start
        return self.config.epsilon_start + progress * span

    def act(
        self,
        state: NDArray[np.floating],
        exploration: float = 0.0,
        random_source: random.Random | None = None,
    ) -> int:
        """Choose an epsilon-greedy action for one state vector."""
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be between zero and one")
        source = random_source or random
        if exploration > 0.0 and source.random() < exploration:
            return source.randrange(self.action_size)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            values = self.online(state_tensor)
        return int(values.argmax(dim=1).item())

    def update(self) -> float | None:
        """Perform one replay update and return its Huber loss when ready."""
        ready_size = max(self.config.minimum_replay_size, self.config.batch_size)
        if len(self.replay) < ready_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(batch.terminals, dtype=torch.float32, device=self.device)

        current_values = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions)
            targets = rewards + self.config.gamma * next_values * (1.0 - terminals)

        loss = nn.functional.smooth_l1_loss(current_values, targets)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        self.optimiser.step()
        self._soft_update()
        self.updates += 1
        return float(loss.item())

    def _soft_update(self) -> None:
        """Move target weights toward online weights by the configured fraction."""
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target.parameters(), self.online.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(online_parameter, alpha=self.config.tau)

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        """Persist weights, optimiser state, configuration, and run metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state": self.online.state_dict(),
                "target_state": self.target.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "config": self.config.as_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: Path) -> dict[str, Any]:
        """Safely restore a restricted weights-only checkpoint and return its metadata."""
        payload = self.read_checkpoint(path, self.device)
        return self.restore(payload)

    @staticmethod
    def read_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
        """Read a checkpoint through PyTorch's restricted weights-only loader."""
        payload = torch.load(path, map_location=device, weights_only=True)
        required = {"online_state", "target_state", "optimiser_state", "config"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError("checkpoint does not match the expected Double DQN format")
        return payload

    def restore(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Restore network and optimiser tensors from a validated checkpoint mapping."""
        self.online.load_state_dict(payload["online_state"])
        self.target.load_state_dict(payload["target_state"])
        self.optimiser.load_state_dict(payload["optimiser_state"])
        return dict(payload.get("metadata", {}))
