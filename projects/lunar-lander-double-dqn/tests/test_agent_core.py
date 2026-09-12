"""Focused tests for deterministic Double DQN components."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from lunar_dqn.agent import DoubleDQNAgent
from lunar_dqn.config import DQNConfig
from lunar_dqn.replay import ReplayBuffer
from lunar_dqn.training import bootstrap_terminal, learning_reward


def compact_config(**changes) -> DQNConfig:
    """Return a small valid configuration for unit tests."""
    values = {
        "batch_size": 2,
        "replay_capacity": 8,
        "minimum_replay_size": 2,
        "hidden_size": 8,
    }
    values.update(changes)
    return DQNConfig(**values)


def test_replay_buffer_wraps_and_samples_expected_shapes() -> None:
    """The circular buffer should retain capacity rows with typed batch arrays."""
    replay = ReplayBuffer(capacity=3, state_size=2, seed=7)
    for index in range(5):
        state = np.array([index, index + 1], dtype=np.float32)
        replay.add(state, index % 2, float(index), state + 1, index == 4)
    batch = replay.sample(3)
    assert len(replay) == 3
    assert batch.states.shape == (3, 2)
    assert batch.actions.dtype == np.int64
    assert set(batch.rewards.ravel()) == {2.0, 3.0, 4.0}


def test_epsilon_schedule_reaches_configured_floor() -> None:
    """Exploration should decay linearly and remain at its configured floor."""
    config = compact_config(epsilon_decay_episodes=10)
    agent = DoubleDQNAgent(2, 2, config, device=torch.device("cpu"))
    assert agent.epsilon(0) == pytest.approx(1.0)
    assert agent.epsilon(5) == pytest.approx(0.525)
    assert agent.epsilon(20) == pytest.approx(0.05)


def test_terminal_transitions_do_not_bootstrap() -> None:
    """A terminal transition should update from reward without a future-value term."""
    config = compact_config(gamma=0.99, tau=1.0)
    agent = DoubleDQNAgent(2, 2, config, device=torch.device("cpu"))
    for parameter in agent.online.parameters():
        parameter.data.zero_()
    agent.target.load_state_dict(agent.online.state_dict())
    state = np.zeros(2, dtype=np.float32)
    agent.replay.add(state, 0, 1.0, state, True)
    agent.replay.add(state, 1, 1.0, state, True)
    loss = agent.update()
    assert loss is not None
    assert loss == pytest.approx(0.5, rel=1e-4)


def test_double_dqn_selects_online_action_and_uses_target_value() -> None:
    """The online argmax should index the target network rather than take its maximum."""
    config = compact_config(gamma=1.0)
    agent = DoubleDQNAgent(2, 2, config, device=torch.device("cpu"))
    for parameter in agent.online.parameters():
        parameter.data.zero_()
    for parameter in agent.target.parameters():
        parameter.data.zero_()
    agent.online.layers[-1].bias.data.copy_(torch.tensor([0.0, 2.0]))
    agent.target.layers[-1].bias.data.copy_(torch.tensor([10.0, 3.0]))
    state = np.zeros(2, dtype=np.float32)
    agent.replay.add(state, 0, 0.0, state, False)
    agent.replay.add(state, 1, 0.0, state, False)
    assert agent.update() == pytest.approx(1.5, rel=1e-4)


def test_greedy_action_is_independent_of_random_source() -> None:
    """Zero exploration should always use the largest network output."""
    agent = DoubleDQNAgent(2, 2, compact_config(), device=torch.device("cpu"))
    for parameter in agent.online.parameters():
        parameter.data.zero_()
    final_layer = agent.online.layers[-1]
    final_layer.bias.data.copy_(torch.tensor([-1.0, 2.0]))
    source = random.Random(1)
    before = source.getstate()
    action = agent.act(np.zeros(2, dtype=np.float32), 0.0, source)
    assert action == 1
    assert source.getstate() == before


def test_reward_shaping_can_be_disabled() -> None:
    """The raw environment score should pass through when shaping is disabled."""
    state = np.zeros(8, dtype=np.float32)
    config = compact_config(use_reward_shaping=False)
    assert learning_reward(3.25, state, False, config) == 3.25


def test_crash_with_two_contacts_does_not_receive_touchdown_bonus():
    state = np.zeros(8, dtype=np.float32)
    state[6:] = 1
    assert learning_reward(-100.0, state, True, compact_config()) == -100.0


def test_truncated_terminal_does_not_receive_touchdown_bonus():
    state = np.zeros(8, dtype=np.float32)
    state[6:] = 1
    assert learning_reward(100.0, state, True, compact_config(), truncated=True) == 100.0


@pytest.mark.parametrize(
    "reward,terminated,truncated,x,contacts,want",
    [
        (-100.0, True, False, 0.0, True, 0.0),
        (100.0, True, False, 0.0, True, 1.0),
        (100.0, False, True, 0.0, True, 0.0),
        (100.0, True, False, 0.2, True, 0.0),
        (100.0, True, False, 0.0, False, 0.0),
        (1.0, False, False, 0.0, True, 0.0),
    ],
)
def test_evaluation_distinguishes_success_from_crash_and_time_limit(
    monkeypatch,
    reward,
    terminated,
    truncated,
    x,
    contacts,
    want,
):
    from lunar_dqn import training

    class Environment:
        def reset(self, *, seed):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            state = np.zeros(8, dtype=np.float32)
            state[0] = x
            state[6:] = contacts
            return state, reward, terminated, truncated, {}

        def close(self):
            pass

    class Agent:
        config = compact_config()

        def act(self, state, exploration):
            return 0

    monkeypatch.setattr(training, "make_environment", lambda *args: Environment())
    result = training.evaluate(Agent(), compact_config(max_steps=1), episodes=1)
    assert result["centred_landing_rate"] == want
    assert result["rewards"] == [reward]
    assert result["episode_outcomes"][0]["final_raw_reward"] == reward
    assert result["episode_outcomes"][0]["centred_landing"] == bool(want)
    assert result["episode_outcomes"][0]["seed"] == 10532


def test_evaluation_rejects_training_seed_overlap_before_environment_creation(monkeypatch):
    from lunar_dqn import training

    agent = DoubleDQNAgent(8, 4, compact_config(episodes=10_000), device=torch.device("cpu"))
    monkeypatch.setattr(training, "make_environment", lambda *args: pytest.fail("seed overlap"))
    with pytest.raises(ValueError, match="overlap"):
        training.evaluate(agent, compact_config(), episodes=1)


def test_training_persists_update_losses_outcomes_and_checkpoint_identity(tmp_path, monkeypatch):
    import csv
    import hashlib
    import json
    from types import SimpleNamespace

    from lunar_dqn import training

    class Environment:
        observation_space = SimpleNamespace(shape=(8,))
        action_space = SimpleNamespace(n=4)

        def reset(self, *, seed):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            state = np.zeros(8, dtype=np.float32)
            state[6:] = 1
            return state, -100.0, True, False, {}

        def close(self):
            pass

    monkeypatch.setattr(training, "make_environment", lambda *args: Environment())
    monkeypatch.setattr("lunar_dqn.agent.select_device", lambda: torch.device("cpu"))
    _, records = training.train(compact_config(episodes=2, train_every=1), tmp_path)
    assert records[-1].outcome == "crash"
    updates = list(csv.DictReader((tmp_path / "update_losses.csv").open()))
    assert len(updates) == 1
    assert np.isfinite(float(updates[0]["loss"]))
    assert int(updates[0]["episode"]) == 2
    metadata = json.loads((tmp_path / "training_metadata.json").read_text())
    checkpoint = tmp_path / "double_dqn_checkpoint.pt"
    assert metadata["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["first_training_seed"] == 533
    assert metadata["last_training_seed"] == 534


def test_time_limit_truncation_keeps_bootstrap_target() -> None:
    """A time limit ends the rollout without changing the underlying task terminal state."""
    assert bootstrap_terminal(terminated=False, truncated=True) is False
    assert bootstrap_terminal(terminated=True, truncated=False) is True


def test_checkpoint_round_trip_uses_expected_safe_format(tmp_path) -> None:
    """Saved tensor state should restore without unrestricted object loading."""
    config = compact_config()
    original = DoubleDQNAgent(2, 2, config, device=torch.device("cpu"))
    checkpoint = tmp_path / "agent.pt"
    original.save(checkpoint, metadata={"kind": "test"})
    payload, checkpoint_sha256 = DoubleDQNAgent.read_checkpoint_with_digest(checkpoint)
    assert len(checkpoint_sha256) == 64
    assert set(checkpoint_sha256) <= set("0123456789abcdef")
    restored = DoubleDQNAgent(2, 2, config, device=torch.device("cpu"))
    assert restored.restore(payload) == {"kind": "test"}
    for expected, actual in zip(
        original.online.parameters(), restored.online.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "changes",
    [
        {"epsilon_decay_episodes": 0},
        {"learning_rate": 0.0},
        {"minimum_replay_size": 9, "replay_capacity": 8},
    ],
)
def test_invalid_configurations_are_rejected(changes) -> None:
    """Invalid schedules and replay bounds should fail before a run starts."""
    with pytest.raises(ValueError):
        compact_config(**changes).validate()
