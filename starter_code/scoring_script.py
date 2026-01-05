#!/usr/bin/env python3
"""
scoring_script.py
Organizer-use only: compute accuracy and macro F1 for submissions
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python scoring_script.py <submission_csv>")
        sys.exit(1)

    submission_file = sys.argv[1]

    if not os.path.exists(submission_file):
        print(f"Submission file {submission_file} does not exist.")
        sys.exit(1)

    if not os.path.exists("data/test_labels.csv"):
        print("Ground-truth file data/test_labels.csv not found. Organizers only.")
        sys.exit(1)

    # Load files
    submission = pd.read_csv(submission_file)
    truth = pd.read_csv("data/test_labels.csv")

    # Check matching IDs
    if not submission['graph_id'].equals(truth['graph_id']):
        print("Graph IDs in submission do not match ground-truth IDs.")
        sys.exit(1)

    # Compute metrics
    acc = accuracy_score(truth['target'], submission['predicted'])
    f1 = f1_score(truth['target'], submission['predicted'], average='macro')

    print(f"✅ Submission Accuracy: {acc:.4f}")
    print(f"✅ Submission Macro F1: {f1:.4f}")

if __name__ == "__main__":
    main()
