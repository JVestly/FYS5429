# FYS5429
Project in FYS5429
# RL-GNN Decoder for Bivariate Bicycle CSS Codes

This repository contains a notebook-based experiment for training and testing a reinforcement-learning graph neural network decoder for CSS quantum error-correcting codes, with a focus on bivariate bicycle codes.

The code compares an RL-GNN decoder against simpler decoding baselines, including greedy decoding and a projected MWPM baseline. The main workflow is split into training, evaluation, and reusable helper utilities.

## Repository Contents

```text
.
├── utils.py
├── training.ipynb
├── tests.ipynb
└── Figures/

##
utils.py contains the reusable decoding utilities, including syndrome calculation, error sampling, Tanner graph construction, the CSS correction environment, and the RL actor-critic model. training.ipynb trains and selects the RL-GNN decoder configuration using imitation learning and reinforcement learning. tests.ipynb evaluates the trained decoder against baseline methods on BB-code error-correction tasks. Figures/ contains the plots and visual outputs generated from the training and evaluation notebooks.
