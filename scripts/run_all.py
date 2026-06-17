from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command reproduction pipeline.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    if args.mode == "full":
        train_episodes, eval_episodes, pi_episodes, sens_episodes, sens_pi = 1000, 1000, 1000, 200, 50
    else:
        train_episodes, eval_episodes, pi_episodes, sens_episodes, sens_pi = 2, 2, 1, 1, 0

    py = sys.executable
    sensitivity_cmd = [
        py,
        "scripts/run_sensitivity.py",
        "--config",
        args.config,
        "--episodes",
        str(sens_episodes),
        "--pi-episodes",
        str(sens_pi),
    ]
    if args.mode == "quick":
        sensitivity_cmd.extend(["--parameters", "bom_cv,delivery_window", "--max-values", "2", "--no-plot"])

    train_cmd = [py, "scripts/train_rlbr.py", "--config", args.config, "--episodes", str(train_episodes)]
    if args.mode == "quick":
        train_cmd.extend(["--epochs", "1", "--regularization-samples", "0", "--rollout-episodes", "1"])

    commands = [
        train_cmd,
        [
            py,
            "scripts/run_benchmark.py",
            "--config",
            args.config,
            "--episodes",
            str(eval_episodes),
            "--pi-episodes",
            str(pi_episodes),
        ],
        sensitivity_cmd,
    ]
    for cmd in commands:
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
