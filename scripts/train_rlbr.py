#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from rl_ato.config import load_config, to_nested_dict
from rl_ato.scenario import ScenarioGenerator, make_instance
from rl_ato.train import train_rlbr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the structure-aware PPO/RLBR model.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--regularization-samples", type=int, default=None)
    parser.add_argument("--rollout-episodes", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument("--best-out", default=None)
    parser.add_argument("--out", default="outputs/rlbr.pt")
    parser.add_argument("--history", default="outputs/train_history.csv")
    args = parser.parse_args()

    exp, ppo, sens = load_config(args.config)
    if args.episodes is not None:
        exp.train_episodes = args.episodes
    if args.epochs is not None:
        ppo.epochs = args.epochs
    if args.network_seed is not None:
        ppo.network_seed = args.network_seed
    if args.regularization_samples is not None:
        ppo.regularization_samples = args.regularization_samples
    if args.rollout_episodes is not None:
        ppo.rollout_episodes = args.rollout_episodes
    instance = make_instance(exp, seed=exp.seed)
    validation_scenarios = None
    if args.validation_episodes > 0:
        validation_generator = ScenarioGenerator(instance, start_episode=2_000_000)
        validation_scenarios = [validation_generator.sample() for _ in range(args.validation_episodes)]
    best_out = args.best_out if args.best_out is not None else args.out
    policy, history = train_rlbr(
        instance,
        exp,
        ppo,
        progress=True,
        validation_scenarios=validation_scenarios,
        validation_interval=args.validation_interval,
        best_path=best_out if validation_scenarios and args.validation_interval > 0 else None,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(out))
    pd.DataFrame(history).to_csv(args.history, index=False)
    used_config = to_nested_dict(exp, ppo, sens)
    Path(out.with_suffix(".config.yaml")).write_text(yaml.safe_dump(used_config, sort_keys=False), encoding="utf-8")
    print(f"saved model: {out}")
    print(f"saved history: {args.history}")


if __name__ == "__main__":
    main()
