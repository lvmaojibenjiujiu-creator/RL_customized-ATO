# RL Customized ATO

Code for the customized assemble-to-order experiments with delayed order-specific BOM revelation, stochastic component lead times, RLBR, and benchmark policies.

## Contents

- `rl_ato/`: simulator, instance generator, policies, RLBR model, PI oracle, evaluation, and plotting.
- `scripts/`: experiment entry points.
- `configs/`: base and formal experiment settings.
- `requirements.txt`: Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

Gurobi is required for the PI oracle and integer OBCA allocation.

## Training

```bash
python3 scripts/train_rlbr.py \
  --config configs/base.yaml \
  --episodes 1000 \
  --out outputs/rlbr.pt
```

Formal tuning:

```bash
python3 scripts/tune_rlbr_formal.py \
  --config configs/formal.yaml \
  --candidate-models outputs/rlbr.pt \
  --validation-episodes 80 \
  --selected-model outputs/rlbr_formal_tuned.pt \
  --selected-config configs/formal_tuned.yaml
```

## Benchmark

```bash
python3 scripts/run_benchmark.py \
  --config configs/formal_tuned.yaml \
  --model outputs/rlbr_formal_tuned.pt \
  --policies RLBR,NVD,DTP,SAA-BS-OBCA \
  --episodes 120 \
  --pi-episodes 120 \
  --dtp-known-scale 1.0 \
  --saa-known-scale 0 \
  --saa-train-episodes 100 \
  --saa-step-sizes 8,4,2,1 \
  --saa-allocation-solver gurobi \
  --summary-out outputs/benchmark_summary.csv \
  --episode-out outputs/benchmark_episodes.csv
```

The PI oracle is an integer Gurobi model. SAA-BS-OBCA uses integer base-stock levels, integer replenishment quantities, and integer Gurobi OBCA allocation.

## Table 4

```bash
python3 scripts/run_table4.py \
  --config configs/formal_tuned.yaml \
  --train-episodes 1000 \
  --eval-episodes 1000 \
  --pi-episodes 50 \
  --scales 5x15,10x20,20x100 \
  --patterns poisson,seasonal \
  --include-saa-bs-obca \
  --saa-train-episodes 100 \
  --saa-step-sizes 8,4,2,1 \
  --saa-allocation-solver gurobi \
  --out-dir outputs/table4
```

## Sensitivity Analysis

```bash
python3 scripts/run_sensitivity.py \
  --config configs/formal_tuned.yaml \
  --model outputs/rlbr_formal_tuned.pt \
  --episodes 200 \
  --pi-episodes 50 \
  --include-saa-bs-obca \
  --saa-train-episodes 100 \
  --saa-step-sizes 8,4,2,1 \
  --saa-allocation-solver gurobi \
  --out outputs/sensitivity_summary.csv \
  --fig-dir outputs/sensitivity_figures \
  --formats png,pdf
```

## Reproducible Instances

```bash
python3 scripts/export_instance.py \
  --config configs/base.yaml \
  --episodes 1000 \
  --out-dir outputs/instance_bundle
```

## Paper Artifacts

```bash
python3 scripts/build_paper_artifacts.py \
  --table4 outputs/table4/table4_summary.csv \
  --sensitivity-summary outputs/sensitivity_summary.csv \
  --sensitivity-fig-dir outputs/sensitivity_figures \
  --out-dir outputs/paper_numerical_results
```
