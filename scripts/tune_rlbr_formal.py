from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from rl_ato.config import load_config, to_nested_dict
from rl_ato.evaluate import benchmark_policies, evaluate_policy
from rl_ato.policies import DTPPolicy, NVDPolicy
from rl_ato.rlbr import RLBRPolicy
from rl_ato.scenario import ScenarioGenerator, make_instance
from rl_ato.train import train_rlbr


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RLBR training/checkpoint/calibration on a fixed validation bench.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out-dir", default="outputs/formal_tuning")
    parser.add_argument("--train-episodes", type=int, default=240)
    parser.add_argument("--validation-episodes", type=int, default=80)
    parser.add_argument("--validation-interval", type=int, default=40)
    parser.add_argument("--network-seeds", default="", help="Comma-separated seeds. Empty uses the config/default seed.")
    parser.add_argument("--skip-training", action="store_true", help="Only calibrate supplied candidate models.")
    parser.add_argument("--candidate-models", default="", help="Extra comma-separated pretrained RLBR checkpoints.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rollout-episodes", type=int, default=None)
    parser.add_argument("--regularization-samples", type=int, default=0)
    parser.add_argument("--base-stock-scale-grid", default="1.0,1.2,1.5")
    parser.add_argument("--base-stock-floor-grid", default="0.0,1.2,1.5")
    parser.add_argument("--base-stock-safety-grid", default="0.0")
    parser.add_argument("--shadow-price-scale-grid", default="1.0")
    parser.add_argument("--selected-model", default="outputs/rlbr_formal_tuned.pt")
    parser.add_argument("--selected-config", default="configs/formal_tuned.yaml")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp, ppo, sens = load_config(args.config)
    exp.train_episodes = args.train_episodes
    exp.eval_episodes = args.validation_episodes
    if args.epochs is not None:
        ppo.epochs = args.epochs
    if args.rollout_episodes is not None:
        ppo.rollout_episodes = args.rollout_episodes
    if args.regularization_samples is not None:
        ppo.regularization_samples = args.regularization_samples

    instance = make_instance(exp, seed=exp.seed)
    validation_generator = ScenarioGenerator(instance, start_episode=2_000_000)
    validation_scenarios = [validation_generator.sample() for _ in range(args.validation_episodes)]
    baselines = [NVDPolicy(instance), DTPPolicy(instance)]
    baseline_episodes, baseline_summary = benchmark_policies(baselines, instance, validation_scenarios)
    baseline_episodes.to_csv(out_dir / "baseline_validation_episodes.csv", index=False)
    baseline_summary.to_csv(out_dir / "baseline_validation_summary.csv", index=False)
    best_baseline_cost = float(baseline_summary["cost_mean"].min())

    candidate_models: list[dict[str, str]] = []
    for model in [x.strip() for x in args.candidate_models.split(",") if x.strip()]:
        candidate_models.append({"label": Path(model).stem, "path": model, "source": "pretrained"})

    if not args.skip_training:
        seeds = _ints(args.network_seeds)
        if not seeds:
            seeds = [int(ppo.network_seed) if int(ppo.network_seed) >= 0 else int(instance.seeds["network_initialization"])]
        for seed in seeds:
            ppo.network_seed = int(seed)
            model_path = out_dir / f"rlbr_seed_{seed}.pt"
            history_path = out_dir / f"history_seed_{seed}.csv"
            print(f"training RLBR seed={seed} for {args.train_episodes} episodes", flush=True)
            policy, history = train_rlbr(
                instance,
                exp,
                ppo,
                progress=True,
                validation_scenarios=validation_scenarios,
                validation_interval=args.validation_interval,
                best_path=str(model_path),
            )
            policy.save(str(model_path))
            pd.DataFrame(history).to_csv(history_path, index=False)
            candidate_models.append({"label": f"seed_{seed}", "path": str(model_path), "source": "trained"})
    if not candidate_models:
        raise ValueError("No candidate models: provide --candidate-models or omit --skip-training.")

    rows = []
    for candidate in candidate_models:
        for scale in _floats(args.base_stock_scale_grid):
            for floor in _floats(args.base_stock_floor_grid):
                for safety in _floats(args.base_stock_safety_grid):
                    for shadow in _floats(args.shadow_price_scale_grid):
                        ppo.base_stock_scale = scale
                        ppo.base_stock_floor = floor
                        ppo.base_stock_safety = safety
                        ppo.shadow_price_scale = shadow
                        policy = RLBRPolicy(instance, ppo)
                        policy.load(candidate["path"])
                        df = evaluate_policy(policy, instance, validation_scenarios)
                        cost = float(df["cost"].mean())
                        dominance_penalty = max(0.0, cost - best_baseline_cost) * 1000.0
                        rows.append(
                            {
                                **candidate,
                                "base_stock_scale": scale,
                                "base_stock_floor": floor,
                                "base_stock_safety": safety,
                                "shadow_price_scale": shadow,
                                "cost_mean": cost,
                                "fill_rate": float(df["fill_rate"].mean()),
                                "ontime_rate": float(df["ontime_rate"].mean()),
                                "residual_inventory_ratio": float(
                                    df["residual_inventory_ratio"].mean()
                                    if "residual_inventory_ratio" in df
                                    else df["mismatch_rate"].mean()
                                ),
                                "mismatch_rate": float(df["mismatch_rate"].mean()),
                                "best_baseline_cost": best_baseline_cost,
                                "score": cost + dominance_penalty,
                                "rlbr_is_best": bool(cost < best_baseline_cost),
                            }
                        )

    result = pd.DataFrame(rows).sort_values(["score", "cost_mean"])
    result.to_csv(out_dir / "rlbr_calibration_grid.csv", index=False)
    best = result.iloc[0].to_dict()
    selected_model = Path(args.selected_model)
    selected_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(best["path"]), selected_model)

    ppo.base_stock_scale = float(best["base_stock_scale"])
    ppo.base_stock_floor = float(best["base_stock_floor"])
    ppo.base_stock_safety = float(best["base_stock_safety"])
    ppo.shadow_price_scale = float(best["shadow_price_scale"])
    tuned_config = to_nested_dict(exp, ppo, sens)
    selected_config = Path(args.selected_config)
    selected_config.parent.mkdir(parents=True, exist_ok=True)
    selected_config.write_text(yaml.safe_dump(tuned_config, sort_keys=False), encoding="utf-8")
    (out_dir / "selected.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(result.head(12).to_string(index=False))
    print(f"selected model: {selected_model}")
    print(f"selected config: {selected_config}")


if __name__ == "__main__":
    main()
