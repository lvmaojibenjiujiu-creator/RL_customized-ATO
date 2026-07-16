# RLBR for customized assemble-to-order planning

This repository implements the RLBR policy and the NVD, DTP, DHP, SAA-OBCA, RH-SPT, and perfect-information benchmarks for customized assemble-to-order planning with delayed bill-of-material information and stochastic replenishment lead times.

The numerical settings are collected in `configs/manuscript.yaml`. The default comparison uses a 40-period horizon, 1,000 RLBR training episodes, and 100 common held-out paths for every policy and the perfect-information benchmark.

## Installation

Python 3.9 and a working Gurobi license are required.

```bash
python -m pip install -r requirements.txt
```

## Training

Train one model for each demand-pattern and scale combination used in the comparison. For example:

```bash
python train.py --products 5 --components 15 --demand-pattern poisson --model-out outputs/models/rlbr_poisson_5x15.pt
```

## Evaluation

With the trained models stored under the naming convention shown above, run:

```bash
python experiment.py --models-dir outputs/models --output-dir outputs/comparison
```

The sensitivity study keeps one baseline RLBR model fixed and recalibrates or resolves the comparison policies for each perturbed instance:

```bash
python sensitivity.py --model outputs/models/rlbr_poisson_5x15.pt --output-dir outputs/sensitivity
```
