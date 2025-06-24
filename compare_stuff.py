#!/usr/bin/env python3
import sys
import pandas as pd

def main(file1: str, file2: str):
    # Load
    df1 = pd.read_csv(file1, usecols=["id","label"])
    df2 = pd.read_csv(file2, usecols=["id","label"])
    
    # Merge on id, keep only matching rows
    df = pd.merge(df1, df2, on="id", how="inner", suffixes=("_1","_2"))
    total = len(df)
    if total == 0:
        print("No matching IDs between the two files.")
        return
    
    # Find mismatches
    mismatches = df[df["label_1"] != df["label_2"]]
    n_diff = len(mismatches)
    
    # Report
    pct = n_diff / total * 100
    print(f"{n_diff} of {total} labels differ  ({pct:.2f}%)\n")
    
    if n_diff:
        print("Differences:")
        for _, row in mismatches.iterrows():
            print(f"  ID {row['id']}: {row['label_1']} → {row['label_2']}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file1.csv> <file2.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
