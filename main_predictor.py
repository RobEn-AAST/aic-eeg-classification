import torch
import numpy as np
import pandas as pd
from modules.competition_dataset import EEGDataset, decode_label, position_decode
from Models import FilterBankRTSClassifier

# Device and public variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT = "test"  #IW
TRAIN_SPLIT = "train"  # split for training
DATA_PATH = "./data/mtcaic3"

# -----------------------------------------------------------------------------
# Motor Imagery pipeline with FilterBankRTSClassifier
# -----------------------------------------------------------------------------
# MI-specific settings (Optuna best)
MI_BATCH_SIZE = 64
MI_WINDOW_LENGTH = 1000
MI_STRIDE = 85
MI_TMIN = 60
MI_CHANNELS = ["FZ", "CZ", "PZ", "PO7", "OZ"]
MI_FS = 100
MI_FILTER_ORDER = 3
MI_N_ESTIMATORS = 300
MI_MAX_DEPTH = None
MI_MIN_SAMPLES_SPLIT = 7
MI_MIN_SAMPLES_LEAF = 3
MI_MAX_FEATURES = "sqrt"


def run_mi_task():
    # Build lookup for submission IDs
    df_meta = pd.read_csv(f"{DATA_PATH}/{SPLIT}.csv")
    lookup = {(row.task, str(row.subject_id), str(row.trial_session), str(row.trial)): row.id for row in df_meta.itertuples() if row.task.upper() == "MI"}

    # Load and train classifier
    ds_train = EEGDataset(DATA_PATH, MI_WINDOW_LENGTH, MI_STRIDE, task="MI", split=TRAIN_SPLIT, read_labels=True, data_fraction=1, tmin=MI_TMIN, eeg_channels=MI_CHANNELS)
    X_train = np.stack([x.numpy() for x, _ in ds_train])
    y_train = np.array([label[0] for _, label in ds_train])
    clf = FilterBankRTSClassifier(
        fs=MI_FS,
        order=MI_FILTER_ORDER,
        n_estimators=MI_N_ESTIMATORS,
        max_depth=MI_MAX_DEPTH,
        min_samples_split=MI_MIN_SAMPLES_SPLIT,
        min_samples_leaf=MI_MIN_SAMPLES_LEAF,
        max_features=MI_MAX_FEATURES,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Inference
    ds_inf = EEGDataset(DATA_PATH, MI_WINDOW_LENGTH, MI_STRIDE, task="MI", split=SPLIT, read_labels=False, data_fraction=1, tmin=MI_TMIN, eeg_channels=MI_CHANNELS)
    windows = [x.numpy() for x, _ in ds_inf]
    codes = ds_inf.labels.numpy()
    X_windows = np.stack(windows)
    probs = clf.predict_proba(X_windows)
    results = []
    for code in sorted(np.unique(codes)):
        idxs = np.where(codes == code)[0]
        avg_prob = probs[idxs].mean(axis=0)
        pred = int(np.argmax(avg_prob))
        subj, sess, trial = position_decode(int(code))
        key = ("MI", subj, sess, trial)
        if key not in lookup:
            raise RuntimeError(f"ID not found for MI key {key}")
        results.append({"id": lookup[key], "label": decode_label(pred, "MI")})
    return results


# -----------------------------------------------------------------------------
# SSVEP pipeline with FilterBankRTSClassifier
# -----------------------------------------------------------------------------
# SSVEP-specific settings (Optuna best)
SS_WINDOW_LENGTH = 1000
SS_STRIDE = 85
SS_TMIN = 60
SS_CHANNELS = ["C4", "CZ", "PZ"]
SS_FS = 100
SS_FILTER_ORDER = 3
SS_N_ESTIMATORS = 300
SS_MAX_DEPTH = None
SS_MIN_SAMPLES_SPLIT = 7
SS_MIN_SAMPLES_LEAF = 3


def run_ssvep_task():
    # Build lookup for submission IDs
    df_meta = pd.read_csv(f"{DATA_PATH}/{SPLIT}.csv")
    lookup = {(row.task, str(row.subject_id), str(row.trial_session), str(row.trial)): row.id for row in df_meta.itertuples() if row.task.upper() == "SSVEP"}

    # Load and train classifier
    ds_train = EEGDataset(DATA_PATH, SS_WINDOW_LENGTH, SS_STRIDE, task="SSVEP", split=TRAIN_SPLIT, read_labels=True, data_fraction=1, tmin=SS_TMIN, eeg_channels=SS_CHANNELS)
    X_train = np.stack([x.numpy() for x, _ in ds_train])
    y_train = np.array([label[0] for _, label in ds_train])
    clf = FilterBankRTSClassifier(
        fs=SS_FS,
        order=SS_FILTER_ORDER,
        n_estimators=SS_N_ESTIMATORS,
        max_depth=SS_MAX_DEPTH,
        min_samples_split=SS_MIN_SAMPLES_SPLIT,
        min_samples_leaf=SS_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Inference
    ds_inf = EEGDataset(DATA_PATH, SS_WINDOW_LENGTH, SS_STRIDE, task="SSVEP", split=SPLIT, read_labels=False, data_fraction=1, tmin=SS_TMIN, eeg_channels=SS_CHANNELS)
    windows = [x.numpy() for x, _ in ds_inf]
    codes = ds_inf.labels.numpy()
    X_windows = np.stack(windows)
    probs = clf.predict_proba(X_windows)
    results = []
    for code in sorted(np.unique(codes)):
        idxs = np.where(codes == code)[0]
        avg_prob = probs[idxs].mean(axis=0)
        pred = int(np.argmax(avg_prob))
        subj, sess, trial = position_decode(int(code))
        key = ("SSVEP", subj, sess, trial)
        if key not in lookup:
            raise RuntimeError(f"ID not found for SSVEP key {key}")
        results.append({"id": lookup[key], "label": decode_label(pred, "SSVEP")})
    return results


# -----------------------------------------------------------------------------
# Main: combine and save submission.csv
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mi_results = run_mi_task()
    ssvep_results = run_ssvep_task()
    all_results = mi_results + ssvep_results
    out = "submission.csv"
    pd.DataFrame(all_results).sort_values("id").to_csv(out, index=False)
    print(f"Saved {out}")
