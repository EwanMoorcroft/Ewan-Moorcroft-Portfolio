"""Training, evaluation, and result-serialisation routines."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from lunar_dqn.agent import DoubleDQNAgent, set_global_seed
from lunar_dqn.config import DQNConfig


@dataclass(frozen=True)
class EpisodeRecord:
    """Metrics captured for one training episode."""

    episode: int
    raw_reward: float
    learning_reward: float
    steps: int
    epsilon: float
    elapsed_seconds: float
    seed: int
    outcome: str
    final_raw_reward: float
    terminated: bool
    truncated: bool


def make_environment(environment_id: str, seed: int):
    """Create and seed a discrete-control environment."""
    environment = gym.make(environment_id)
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    return environment


def bootstrap_terminal(terminated: bool, truncated: bool) -> bool:
    """Mask bootstrapping only for true terminal states, not time limits."""
    del truncated
    return bool(terminated)


def centred_landing(raw_reward, state, terminated, truncated, config) -> bool:
    """Identify the v3 successful sleep terminal (+100), excluding crashes (-100)."""
    return bool(
        terminated
        and not truncated
        and raw_reward == 100.0
        and bool(state[6])
        and bool(state[7])
        and abs(float(state[0])) < config.centre_threshold
    )


def episode_outcome(raw_reward, state, terminated, truncated, config) -> str:
    """Record an auditable outcome, including a caller-imposed step limit."""
    if truncated:
        return "truncated"
    if centred_landing(raw_reward, state, terminated, truncated, config):
        return "centred_landing"
    if terminated:
        if raw_reward == -100.0:
            return "crash"
        return "other_success" if raw_reward == 100.0 else "other_terminal"
    return "step_limit"


def learning_reward(
    raw_reward: float,
    next_state: NDArray[np.floating],
    terminal: bool,
    config: DQNConfig,
    *,
    truncated: bool = False,
) -> float:
    """Apply training-only reward shaping while preserving raw scores for reporting."""
    if not config.use_reward_shaping:
        return float(raw_reward)
    x_position, altitude, x_velocity, y_velocity, angle = map(float, next_state[:5])
    left_contact, right_contact = map(bool, next_state[6:8])
    low = altitude < config.low_altitude_threshold
    no_contact = not left_contact and not right_contact
    value = float(raw_reward)
    value -= config.centre_penalty * abs(x_position)
    value -= config.horizontal_velocity_penalty * abs(x_velocity)
    if low:
        value -= config.low_altitude_centre_penalty * abs(x_position)
        if no_contact:
            value -= config.hover_penalty
    centred = abs(x_position) < config.centre_threshold
    upright = abs(angle) < 0.15
    controlled = -0.40 < y_velocity < -0.03
    if low and centred and upright and controlled:
        value += config.controlled_descent_bonus
    if centred_landing(raw_reward, next_state, terminal, truncated, config):
        value += config.centred_touchdown_bonus
    return value


def train(config: DQNConfig, output_dir: Path) -> tuple[DoubleDQNAgent, list[EpisodeRecord]]:
    """Train a Double DQN agent and write a checkpoint plus episode log."""
    config.validate()
    set_global_seed(config.seed)
    environment = make_environment(config.environment_id, config.seed)
    state_size = int(np.prod(environment.observation_space.shape))
    action_size = int(environment.action_space.n)
    agent = DoubleDQNAgent(state_size, action_size, config)
    records: list[EpisodeRecord] = []
    update_losses: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    try:
        for episode in range(config.episodes):
            state, _ = environment.reset(seed=config.seed + episode + 1)
            raw_total = 0.0
            learning_total = 0.0
            exploration = agent.epsilon(episode)
            steps = 0
            for _ in range(config.max_steps):
                action = agent.act(state, exploration=exploration)
                next_state, raw, terminated, truncated, _ = environment.step(action)
                episode_ended = bool(terminated or truncated)
                terminal = bootstrap_terminal(terminated, truncated)
                shaped = learning_reward(raw, next_state, terminal, config, truncated=truncated)
                agent.replay.add(state, action, shaped, next_state, terminal)
                if global_step % config.train_every == 0:
                    loss = agent.update()
                    if loss is not None:
                        update_losses.append(
                            {
                                "update": agent.updates,
                                "global_step": global_step,
                                "episode": episode + 1,
                                "loss": loss,
                            }
                        )
                state = next_state
                raw_total += float(raw)
                learning_total += shaped
                global_step += 1
                steps += 1
                if episode_ended:
                    break
            records.append(
                EpisodeRecord(
                    episode=episode + 1,
                    raw_reward=raw_total,
                    learning_reward=learning_total,
                    steps=steps,
                    epsilon=exploration,
                    elapsed_seconds=time.perf_counter() - started,
                    seed=config.seed + episode + 1,
                    outcome=episode_outcome(raw, state, terminated, truncated, config),
                    final_raw_reward=float(raw),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
    finally:
        environment.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    with (output_dir / "update_losses.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["update", "global_step", "episode", "loss"])
        writer.writeheader()
        writer.writerows(update_losses)
    agent.save(
        output_dir / "double_dqn_checkpoint.pt",
        metadata={
            "episodes": config.episodes,
            "environment_id": config.environment_id,
            "updates": agent.updates,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    write_evaluation(
        {
            "checkpoint_sha256": hashlib.sha256(
                (output_dir / "double_dqn_checkpoint.pt").read_bytes()
            ).hexdigest(),
            "first_training_seed": config.seed + 1,
            "last_training_seed": config.seed + config.episodes,
            "updates": agent.updates,
            "outcome_definition": "LunarLander-v3 terminal reward +100 means success; -100 means crash",
        },
        output_dir / "training_metadata.json",
    )
    return agent, records


def evaluate(
    agent: DoubleDQNAgent,
    config: DQNConfig,
    episodes: int | None = None,
) -> dict[str, Any]:
    """Evaluate a greedy policy on a deterministic sequence of fresh seeds."""
    episode_count = config.evaluation_episodes if episodes is None else episodes
    if episode_count < 1:
        raise ValueError("episodes must be positive")
    config.validate()
    first_seed = config.seed + 10_000
    last_seed = first_seed + episode_count - 1
    training_config = agent.config
    if (
        first_seed <= training_config.seed + training_config.episodes
        and last_seed >= training_config.seed
    ):
        raise ValueError("evaluation seeds overlap the checkpoint's training seeds")
    environment = make_environment(config.environment_id, first_seed)
    rewards: list[float] = []
    lengths: list[int] = []
    outcomes = []
    centred_landings = 0
    try:
        for index in range(episode_count):
            state, _ = environment.reset(seed=config.seed + 10_000 + index)
            total = 0.0
            steps = 0
            for _ in range(config.max_steps):
                action = agent.act(state, exploration=0.0)
                state, reward, terminated, truncated, _ = environment.step(action)
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            rewards.append(total)
            lengths.append(steps)
            if centred_landing(reward, state, terminated, truncated, config):
                centred_landings += 1
            outcomes.append(
                {
                    "episode": index + 1,
                    "seed": first_seed + index,
                    "outcome": episode_outcome(reward, state, terminated, truncated, config),
                    "centred_landing": centred_landing(
                        reward, state, terminated, truncated, config
                    ),
                    "final_raw_reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "final_state": [float(value) for value in state],
                }
            )
    finally:
        environment.close()
    return {
        "environment_id": config.environment_id,
        "episodes": episode_count,
        "mean_reward": float(np.mean(rewards)),
        "standard_deviation_reward": float(np.std(rewards)),
        "minimum_reward": float(np.min(rewards)),
        "maximum_reward": float(np.max(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "centred_landing_rate": centred_landings / episode_count,
        "rewards": rewards,
        "lengths": lengths,
        "episode_outcomes": outcomes,
        "first_environment_seed": first_seed,
        "last_environment_seed": last_seed,
        "training_seed": training_config.seed,
        "training_episodes": training_config.episodes,
    }


def write_evaluation(result: dict[str, Any], path: Path) -> None:
    """Write evaluation metrics as stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
