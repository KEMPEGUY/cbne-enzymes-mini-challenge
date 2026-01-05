# CBNE-ENZYMES Mini-Challenge

## Goal
Outperform the provided hybrid baseline accuracy (~0.2983) on the ENZYMES dataset using a combination of GCNs and CBNE moment features.

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
- Changing the core GNN architecture (e.g., number of layers, type of convolution)

### Provided
- Frozen CBNE moment features (`data/train.csv`, `data/test.csv`) 
- Baseline Hybrid GCN + Moments model (`starter_code/baseline_hybrid.py`) 
- Training / evaluation pipeline (`training/train_eval.py`, `training/run_baseline.py`) 

## How to Run Baseline
```bash
python starter_code/baseline_hybrid.py
```

## Competition Flow

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd gnn-challenge
   ```

2. **Install dependencies**
   ```bash
   pip install -r starter_code/requirements.txt
   ```

3. **Run the baseline** to see hybrid model performance:
   ```bash
   python starter_code/baseline_hybrid.py
   ```

4. **Modify allowed components** (hyperparameters, regularization, fusion strategy, etc.) to improve performance.

5. **Generate predictions** on `test.csv` and optionally submit CSV to organizers.

6. **Scoring (Organizer only)** 
   The official Macro F1 scores are computed using `scoring_script.py` with hidden labels:
   ```bash
   python scoring_script.py submissions/<your_file>.csv
   ```
   > Note: Test labels (`data/test_labels.csv`) are hidden. Participants can compare locally using validation splits, but official ranking is based on hidden labels.

## Submissions / Leaderboard

* Currently, no automated scoring workflow is included. Participants run the baseline, tune allowed components, and compare results locally.
* Optional manual submission: create a CSV with predictions (matching `test.csv` IDs) and share with organizers.

## Winner Determination

The winner of the CBNE-ENZYMES Mini-Challenge will be the participant whose submission achieves the **highest Macro F1 score** on the hidden test set (`data/test_labels.csv`). 

> Note: While participants can validate their models locally using the training/validation split, the official ranking is based solely on the hidden test labels. Accuracy and F1 improvements from including CBNE topological features are expected to provide an advantage.

## Notes (Context Only)

For reference, a standalone 2-layer GCN without CBNE moment features achieves approximately:

* Accuracy: ~0.26
* Macro F1: ~0.21

These results illustrate that including topological features (CBNE moments) improves the prediction model. The official baseline remains the Hybrid GCN + CBNE Moments model.

## Repository Structure

```
gnn-challenge/
│
├─ data/
│   ├─ train.csv
│   ├─ test.csv
│   └─ test_labels.csv (hidden, organizer only)
│
├─ submissions/
│   └─ sample_submission.csv
│
├─ starter_code/
│   ├─ baseline_hybrid.py
│   └─ requirements.txt
│
├─ training/
│   ├─ train_eval.py
│   └─ run_baseline.py
│
├─ scoring_script.py  # Organizer use only
├─ README.md
└─ LICENSE
```

## Tips for Participants

* Dataset is small but non-trivial: topological features matter.
* Aim for creative approaches using allowed modifications.
* Validation F1 may differ from official leaderboard due to hidden test labels.

