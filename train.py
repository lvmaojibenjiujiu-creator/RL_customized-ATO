from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_ato.config import load_config
from rl_ato.scenario import make_instance
from rl_ato.train import train_rlbr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--products", type=int)
    parser.add_argument("--components", type=int)
    parser.add_argument("--demand-pattern", choices=["poisson", "seasonal"])
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--history-out")
    arguments = parser.parse_args()
    experiment, ppo, _benchmarks, _sensitivity = load_config(arguments.config)
    if (arguments.products is None) != (arguments.components is None):
        raise ValueError("products and components must be provided together")
    if arguments.products is not None:
        experiment.products = int(arguments.products)
    if arguments.components is not None:
        experiment.components = int(arguments.components)
    if arguments.demand_pattern is not None:
        experiment.demand_pattern = str(arguments.demand_pattern)
    instance = make_instance(experiment)
    policy, history = train_rlbr(instance, experiment, ppo)
    model_path = Path(arguments.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(model_path))
    if arguments.history_out:
        history_path = Path(arguments.history_out)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
