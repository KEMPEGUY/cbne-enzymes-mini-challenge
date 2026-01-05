Perfect! Here’s the **full updated README** with the new `Notes (Context Only)` section included at the end:

``markdown
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
bash
python starter_code/baseline_hybrid.py
``

## Submissions / Leaderboard

Currently, no automated scoring workflow is included. Participants can run the baseline, tune allowed components, and compare results locally. An automated leaderboard may be added later.

### Optional Manual Submission

Participants can create a CSV with their predictions (matching `test.csv` IDs) and share it with the organizer for local scoring comparison.

## Notes (Context Only)

For reference, a standalone 2-layer GCN without CBNE moment features achieves approximately:

* Accuracy: ~0.26
* Macro F1: ~0.21

These results are for context only to illustrate that including topological features (CBNE moments) improves the prediction model. The official baseline remains the Hybrid GCN + CBNE Moments model.





















