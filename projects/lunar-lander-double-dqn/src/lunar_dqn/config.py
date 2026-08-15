"""Configuration for Double DQN training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for a complete LunarLander run."""

    seed: int = 532
    environment_id: str = "LunarLander-v3"
    episodes: int = 1_000
    max_steps: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    minimum_replay_size: int = 1_000
    train_every: int = 4
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 850
    hidden_size: int = 128
    gradient_clip_norm: float = 10.0
    evaluation_episodes: int = 10
    use_reward_shaping: bool = True
    centre_penalty: float = 0.30
    low_altitude_centre_penalty: float = 2.50
    horizontal_velocity_penalty: float = 0.20
    hover_penalty: float = 0.05
    low_altitude_threshold: float = 0.18
    centred_touchdown_bonus: float = 8.0
    centre_threshold: float = 0.08
    controlled_descent_bonus: float = 0.03

    def validate(self) -> None:
        """Raise a clear error when hyperparameters are internally inconsistent."""
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.episodes < 1 or self.max_steps < 1:
            raise ValueError("episodes and max_steps must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.minimum_replay_size < self.batch_size:
            raise ValueError("minimum_replay_size must be at least batch_size")
        if self.minimum_replay_size > self.replay_capacity:
            raise ValueError("minimum_replay_size must not exceed replay_capacity")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in the interval (0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_episodes < 1:
            raise ValueError("epsilon_decay_episodes must be positive")
        if min(self.train_every, self.hidden_size, self.evaluation_episodes) < 1:
            raise ValueError(
                "training interval, hidden size, and evaluation count must be positive"
            )
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        penalties = (
            self.centre_penalty,
            self.low_altitude_centre_penalty,
            self.horizontal_velocity_penalty,
            self.hover_penalty,
            self.centred_touchdown_bonus,
            self.controlled_descent_bonus,
        )
        if any(value < 0.0 for value in penalties):
            raise ValueError("reward-shaping magnitudes must be non-negative")
        if self.low_altitude_threshold <= 0.0 or self.centre_threshold <= 0.0:
            raise ValueError("reward-shaping thresholds must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable configuration mapping."""
        return asdict(self)
