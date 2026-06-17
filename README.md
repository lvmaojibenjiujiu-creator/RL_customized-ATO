# RL Customized ATO

This repository contains simulation and experiment code for customized assemble-to-order material requirement planning with incomplete order-specific BOM information and stochastic component lead times.

## Structure

- `rl_ato/`: simulator, scenario generator, evaluation tools, RLBR model, and PI oracle.
- `benchmarks/`: DHP-SBR benchmark implementation.
- `scripts/`: training, benchmarking, sensitivity analysis, and paper-artifact entry points.
- `configs/`: base and formal experiment settings.
- `tests/`: lightweight metric checks.

## Setup

```bash
pip install -r requirements.txt
```

Gurobi is required for the PI oracle and integer OBCA allocation.

## Checks

```bash
python3 -m pytest tests/test_metrics.py
python3 scripts/smoke_test_dhp_sbr.py
```

## Main Entry Points

```bash
python3 scripts/train_rlbr.py --config configs/base.yaml --episodes 1000 --out outputs/rlbr.pt
python3 scripts/run_table4.py --config configs/formal_tuned.yaml --out-dir outputs/table4
python3 scripts/run_sensitivity.py --config configs/formal_tuned.yaml --model outputs/rlbr.pt --out outputs/sensitivity_summary.csv
```

Long rolling-horizon runs can be resumed with `scripts/run_rh_spt_checkpoint.py`. Paper tables and figures can be rebuilt with `scripts/build_paper_artifacts.py` after benchmark and sensitivity outputs are available.
