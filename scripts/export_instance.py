#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from rl_ato.config import load_config
from rl_ato.scenario import ScenarioGenerator, make_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a reproducible ATO instance bundle.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs/instance_bundle")
    args = parser.parse_args()

    exp, _ppo, _sens = load_config(args.config)
    instance = make_instance(exp, seed=exp.seed)
    generator = ScenarioGenerator(instance)
    scenarios = [generator.sample(e) for e in range(args.episodes)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "global_parameters": {
            "I": instance.I,
            "J": instance.J,
            "T": instance.T,
            "mean_demand": exp.mean_demand,
            "demand_cv": exp.demand_cv,
            "seasonal_beta": exp.seasonal_beta,
            "seasonal_cycle": exp.seasonal_cycle,
            "component_commonality": exp.component_commonality,
            "realized_commonality": instance.realized_commonality,
            "bom_cv_target": exp.bom_cv,
            "design_lead_time": exp.design_lead_time,
            "delivery_window": exp.delivery_window,
            "min_replenishment_lead_time": exp.min_replenishment_lead_time,
            "max_replenishment_lead_time": exp.max_replenishment_lead_time,
            "backorder_to_holding": exp.backorder_to_holding,
            "demand_correlation_target": exp.demand_correlation,
            "latent_demand_correlation": instance.latent_demand_correlation,
            "empirical_demand_correlation": instance.empirical_demand_correlation,
            "exported_episodes": args.episodes,
        },
        "seeds": instance.seeds,
        "cost_rule": instance.config["cost_rule"],
        "episode_realized_bom_cv": [float(s.realized_bom_cv) for s in scenarios],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "arrays.npz",
        support=instance.support,
        template_bom=instance.template_bom,
        target_common_components=instance.target_common_components,
        demand_lambdas=instance.demand_lambdas,
        seasonal_phases=instance.seasonal_phases,
        holding_costs=instance.holding_costs,
        ordering_costs=instance.ordering_costs,
        backlog_costs=instance.backlog_costs,
        initial_inventory=instance.initial_inventory,
        demand_paths=np.stack([s.demand for s in scenarios]),
        realized_bom_paths=np.stack([s.realized_bom for s in scenarios]),
        replenishment_lead_times=np.stack([s.lead_times for s in scenarios]),
    )
    print(f"saved metadata: {out / 'metadata.json'}")
    print(f"saved arrays: {out / 'arrays.npz'}")


if __name__ == "__main__":
    main()
