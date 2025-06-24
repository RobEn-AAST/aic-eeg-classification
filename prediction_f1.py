#!/usr/bin/env python3
import pandas as pd
from sklearn.metrics import accuracy_score

split = "validation"
PRED_CSV = "./submission.csv"
TRUE_CSV = f"./data/mtcaic3/{split}.csv"

def main():
    # load both, keep task on the ground truth
    df_true = pd.read_csv(TRUE_CSV, usecols=["id","task","label"])
    df_pred = pd.read_csv(PRED_CSV, usecols=["id","label"]).rename(columns={"label":"pred"})
    df = df_true.merge(df_pred, on="id", how="inner")
    if df.empty:
        raise ValueError("No matching IDs found between prediction and truth.")

    acc = {}
    for task in ["MI","SSVEP"]:
        sub = df[df.task == task]
        acc[task] = accuracy_score(sub.label, sub.pred)
        print(f"{task} accuracy: {acc[task]:.4f}")

    combined = (acc["MI"] + acc["SSVEP"]) / 2
    print(f"\nCombined (½·(MI+SSVEP)): {combined:.4f}")

if __name__ == "__main__":
    main()
