from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.dhp_sbr import DHPSBRPolicy
from benchmarks.dhp_sbr_calibration import calibrate_dhp_sbr_tables
from rl_ato.config import ExperimentConfig
from rl_ato.env import ATOEnv
from rl_ato.scenario import Scenario, ScenarioGenerator, make_instance


def _toy_instance():
    cfg = replace(
        ExperimentConfig(),
        seed=2026,
        products=2,
        components=3,
        horizon=5,
        mean_demand=2.0,
        design_lead_time=2,
        delivery_window=1,
        min_replenishment_lead_time=1,
        max_replenishment_lead_time=2,
        family_degree_min=1,
        family_degree_max=2,
        initial_inventory_factor=1.0,
        correlation_pilot_samples=1000,
        train_episodes=2,
        eval_episodes=2,
        pi_episodes=0,
    )
    return make_instance(cfg, seed=cfg.seed)


def _fast_cfg(cache_dir: str, cache_tables: bool = True):
    return {
        "calibration_mode": "formula_init",
        "Z_max": 5,
        "cache_dir": cache_dir,
        "cache_tables": cache_tables,
        "n_cal_paths_small": 2,
        "T_cal": 8,
        "warmup": 2,
    }


def test_no_peeking() -> None:
    inst = _toy_instance()
    gen = ScenarioGenerator(inst, seed=inst.seeds["master"])
    scenario = gen.sample(episode=0)
    altered = Scenario(
        demand=scenario.demand.copy(),
        realized_bom=scenario.realized_bom.copy(),
        lead_times=scenario.lead_times.copy(),
        realized_bom_cv=scenario.realized_bom_cv,
        episode=scenario.episode,
    )
    altered.realized_bom[:, 0, :] += 7.0
    with tempfile.TemporaryDirectory() as tmp:
        policy = DHPSBRPolicy(inst, calibration_config=_fast_cfg(tmp, cache_tables=False), seed=11)
        env_a = ATOEnv(inst)
        obs_a = env_a.reset(scenario)
        env_b = ATOEnv(inst)
        obs_b = env_b.reset(altered)
        act_a = policy.act(env_a, obs_a)
        act_b = policy.act(env_b, obs_b)
    assert np.allclose(act_a.orders, act_b.orders)
    assert act_a.allocations == act_b.allocations


def test_action_feasibility_and_tiny_run() -> None:
    inst = _toy_instance()
    scenario = ScenarioGenerator(inst).sample(episode=1)
    with tempfile.TemporaryDirectory() as tmp:
        policy = DHPSBRPolicy(inst, calibration_config=_fast_cfg(tmp, cache_tables=False), seed=12)
        env = ATOEnv(inst)
        obs = env.reset(scenario)
        done = False
        while not done:
            action = policy.act(env, obs)
            assert np.all(np.asarray(action.orders) >= -1e-9)
            consumption = np.zeros(inst.J, dtype=float)
            for p, s, qty in action.allocations:
                assert env.revealed[int(p), int(s)]
                consumption += scenario.realized_bom[int(p), int(s)] * float(qty)
            assert np.all(consumption <= env.inventory + 1e-9)
            obs, _reward, done, _info = env.step(action)
            if obs is None:
                break
        assert np.isfinite(env.metrics()["cost"])


def test_monotonicity_and_cache() -> None:
    inst = _toy_instance()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fast_cfg(tmp, cache_tables=True)
        first = calibrate_dhp_sbr_tables(inst, cfg, seed=13)
        second = calibrate_dhp_sbr_tables(inst, cfg, seed=13)
        for edge, S in first.S_table.items():
            R = first.R_table[edge]
            assert np.all(np.diff(S) >= 0)
            assert np.all(np.diff(R) <= 0)
            assert np.array_equal(S, second.S_table[edge])
            assert np.array_equal(R, second.R_table[edge])


def main() -> None:
    test_no_peeking()
    test_action_feasibility_and_tiny_run()
    test_monotonicity_and_cache()
    print("DHP-SBR smoke tests passed")


if __name__ == "__main__":
    main()
