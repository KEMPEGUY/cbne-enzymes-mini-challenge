# CBNE–ENZYMES Mini Challenge

## Problem
Graph classification on the **ENZYMES** dataset (6 classes).

Each graph is represented by:
- a graph structure (GCN)
- precomputed **CBNE moment-based topological features**

## Goal
**Outperform the provided hybrid baseline accuracy (~0.2983).**

## Rules
### Allowed
- Modify the GNN architecture
- Change fusion strategy
- Tune hyperparameters
- Add regularization / normalization

### Forbidden
- Modifying CBNE feature generation
- Using external data
- Changing train/test CSV splits

## Provided
- Frozen CBNE moment features
- Baseline Hybrid GCN + Moments model
- Training / evaluation pipeline

## How to Run Baseline
```bash
python training/run_baseline.py

