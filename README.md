# CBNE-ENZYMES Mini-Challenge

## Goal
Outperform the provided hybrid baseline accuracy (~0.2983) on the ENZYMES dataset.

## Rules

### Allowed
- Tune hyperparameters (learning rate, batch size, number of epochs) 
- Add or modify regularization / normalization (dropout, weight decay, etc.) 
- Tune hyperparameters of the CBNE moment features (e.g., number of moments, max power, filtration parameters) 
- Optionally modify the *fusion strategy* (how GCN features and CBNE moment features are combined)

### Forbidden
- Modifying CBNE feature generation code 
- Using external data 
- Changing train/test CSV splits 

### Provided
- Frozen CBNE moment features (`data/train.csv`, `data/test.csv`) 
- Baseline Hybrid GCN + Moments model (`starter_code/baseline_hybrid.py`) 
- Training / evaluation pipeline (`training/train_eval.py`, `training/run_baseline.py`) 

## How to Run Baseline
```bash
python starter_code/baseline_hybrid.py

