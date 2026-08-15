"""Command-line entry point for training and greedy evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from lunar_dqn.agent import DoubleDQNAgent, select_device
from lunar_dqn.config import DQNConfig
from lunar_dqn.training import evaluate, make_environment, train, write_evaluation


def _agent_for_checkpoint(
    checkpoint: Path,
    evaluation_seed: int,
) -> tuple[DoubleDQNAgent, DQNConfig]:
    """Rebuild an agent from saved settings and set the evaluation seed."""
    device = select_device()
    payload = DoubleDQNAgent.read_checkpoint(checkpoint, device)
    saved_config = DQNConfig(**payload["config"])
    saved_config.validate()
    config = replace(saved_config, seed=evaluation_seed)
    environment = make_environment(config.environment_id, config.seed)
    try:
        state_size = int(np.prod(environment.observation_space.shape))
        action_size = int(environment.action_space.n)
    finally:
        environment.close()
    agent = DoubleDQNAgent(state_size, action_size, config, device=device)
    agent.restore(payload)
    return agent, config


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and save a local checkpoint")
    train_parser.add_argument("--output", type=Path, default=Path("runs/default"))
    train_parser.add_argument("--episodes", type=int, default=1_000)
    train_parser.add_argument("--seed", type=int, default=532)
    train_parser.add_argument("--no-reward-shaping", action="store_true")

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a trusted local checkpoint")
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    evaluate_parser.add_argument("--episodes", type=int, default=10)
    evaluate_parser.add_argument("--seed", type=int, default=532)
    return parser


def main() -> int:
    """Run the selected command."""
    args = build_parser().parse_args()
    config = DQNConfig(seed=args.seed)
    if args.command == "train":
        config = replace(
            config,
            episodes=args.episodes,
            use_reward_shaping=not args.no_reward_shaping,
        )
        train(config, args.output)
        print(f"Training outputs written to {args.output}")
        return 0
    agent, config = _agent_for_checkpoint(args.checkpoint, args.seed)
    result = evaluate(agent, config, episodes=args.episodes)
    write_evaluation(result, args.output)
    print(f"Evaluation written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
