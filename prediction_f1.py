#!/usr/bin/env python3
"""
compare_f1.py

Compute overall and per-class F1 scores and display a confusion matrix heatmap
between your predictions and the ground-truth labels.
"""

import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ─── CONFIGURE THESE PATHS ─────────────────────────────────────────────────────
PRED_CSV = "./submission.csv"
TRUE_CSV = "./data/mtcaic3/validation.csv"
# ────────────────────────────────────────────────────────────────────────────────

def load_and_merge(pred_path: str, true_path: str) -> pd.DataFrame:
    # Load
    df_pred = pd.read_csv(pred_path, usecols=["id", "label"])
    df_true = pd.read_csv(true_path, usecols=["id", "label"])
    # Rename true label column so we don’t clash
    df_true = df_true.rename(columns={"label": "label_true"})
    # Merge on id
    df = pd.merge(df_true, df_pred, on="id", how="inner")
    if len(df) == 0:
        raise ValueError("No matching IDs found between the two files.")
    return df

def compute_f1(df: pd.DataFrame):
    y_true = df["label_true"]
    y_pred = df["label"]
    
    # overall (weighted) F1
    f1_overall = f1_score(y_true, y_pred, average="weighted")
    print(f"Overall weighted F1 score: {f1_overall:.4f}\n")
    
    # full per-class report
    print("Per-class precision / recall / F1:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # --- Confusion Matrix Visualization ---
    print('Displaying Confusion Matrix Heatmap...')
    cm = confusion_matrix(y_true, y_pred)
    
    # Get the sorted list of unique labels present in the data
    class_labels = sorted(list(y_true.unique()))
    
    # Create the heatmap plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    
    # Add labels and a title
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Display the plot
    plt.show()


if __name__ == "__main__":
    df = load_and_merge(PRED_CSV, TRUE_CSV)
    compute_f1(df)