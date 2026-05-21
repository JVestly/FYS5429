# FYS5429
Project in FYS5429
# RL-GNN Quantum Error Correcting Decoder for BB codes

This repository contains a notebook-based experiment for training and testing a reinforcement-learning graph neural network decoder for CSS quantum error-correcting codes, with a focus on bivariate bicycle codes.

The code compares an RL-GNN decoder against simpler decoding baselines, including greedy decoding and a projected MWPM baseline. The main workflow is split into training, evaluation, and reusable helper utilities.

## Repository Contents

```text
.
├── utils.py
├── training.ipynb
├── tests.ipynb
└── Figures/
```

`utils.py` contains the reusable decoding utilities, including syndrome calculation, error sampling, Tanner graph construction, the RL environment, and decoder helper functions.

`training.ipynb` contains the training workflow for the RL–GNN decoder.

`tests.ipynb` contains evaluation experiments and baseline comparisons.

`Figures/` contains generated plots used in the report.
