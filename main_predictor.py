import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from modules.competition_dataset import EEGDataset, decode_label, position_decode
from sklearn.metrics import accuracy_score, classification_report
from Models import FilterBankRTSClassifier  # assume this is defined in module

# Device and common settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT = "validation"           # public variable for data split (e.g., "train", "validation", "test")
TRAIN_SPLIT = "train"          # split used for training MI model
data_path = "./data/mtcaic3"
confidence_exponent = 2.0

# -----------------------------------------------------------------------------
# Motor Imagery pipeline with FilterBankRTSClassifier
# -----------------------------------------------------------------------------
# MI-specific data settings
MI_BATCH_SIZE = 64
MI_WINDOW_LENGTH = 250   # from Optuna best trial
MI_STRIDE = 250          # as tuned
MI_TMIN = 0
MI_CHANNELS = ['FZ', 'CZ', 'PZ', 'PO7', 'OZ']
# Classifier hyperparameters from Optuna
MI_FS = 125
MI_FILTER_ORDER = 3
MI_N_ESTIMATORS = 200
MI_MAX_DEPTH = None
MI_MIN_SAMPLES_SPLIT = 6
MI_MIN_SAMPLES_LEAF = 2
MI_MAX_FEATURES = 'sqrt'


def run_mi_task():
    """Train and infer on the Motor Imagery task and return a list of {id, label}."""
    # Build lookup for MI
    df_csv = pd.read_csv(f"{data_path}/{SPLIT}.csv")
    lookup = {
        (row.task, str(row.subject_id), str(row.trial_session), str(row.trial)): row.id
        for row in df_csv.itertuples() if row.task == "MI"
    }

    # Load train and val EEG data
    ds_train = EEGDataset(
        data_path,
        window_length=MI_WINDOW_LENGTH,
        stride=MI_STRIDE,
        task="MI",
        split=TRAIN_SPLIT,
        data_fraction=1,
        tmin=MI_TMIN,
        eeg_channels=MI_CHANNELS,
    )
    X_train = np.stack([x.numpy() for x, _ in ds_train])
    y_train = np.array([label[0] for _, label in ds_train])

    ds_val = EEGDataset(
        data_path,
        window_length=MI_WINDOW_LENGTH,
        stride=MI_STRIDE,
        task="MI",
        split=SPLIT,
        data_fraction=1,
        tmin=MI_TMIN,
        eeg_channels=MI_CHANNELS,
    )
    X_val = np.stack([x.numpy() for x, _ in ds_val])
    y_val = np.array([label[0] for _, label in ds_val])

    # Initialize and fit classifier
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

    # Predict on validation set
    y_pred = clf.predict(X_val)

    # Optionally print metrics
    val_acc = accuracy_score(y_val, y_pred)
    print(f"MI Validation accuracy: {val_acc:.4f}")
    print(classification_report(y_val, y_pred))

    # Prepare submission entries
    results = []
    codes = ds_val.labels.numpy()
    for code, pred in zip(codes, y_pred):
        subj, sess, trial = position_decode(int(code))
        key = ("MI", subj, sess, trial)
        csv_id = lookup.get(key)
        if csv_id is not None:
            results.append({"id": csv_id, "label": decode_label(int(pred), "MI")})
    return results

# -----------------------------------------------------------------------------
# SSVEP pipeline with constant empty predictions
# -----------------------------------------------------------------------------
SSVEP_BATCH_SIZE = 64
SSVEP_WINDOW_LENGTH = 512
SSVEP_STRIDE = SSVEP_WINDOW_LENGTH // 4


def run_ssvep_task():
    """Generate SSVEP submissions with empty labels for each trial ID."""
    # Build lookup for SSVEP
    df_csv = pd.read_csv(f"{data_path}/{SPLIT}.csv")
    lookup = {
        (row.task, str(row.subject_id), str(row.trial_session), str(row.trial)): row.id
        for row in df_csv.itertuples() if row.task == "SSVEP"
    }

    # Iterate through every SSVEP trial code, assign empty label
    results = []
    for key, csv_id in lookup.items():
        results.append({"id": csv_id, "label": ""})
    return results

# -----------------------------------------------------------------------------
# Main execution: combine and save both tasks
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mi_results = run_mi_task()
    ssvep_results = run_ssvep_task()

    # Combine and save to CSV
    all_results = mi_results + ssvep_results
    out = "submission.csv"
    pd.DataFrame(all_results).sort_values("id").to_csv(out, index=False)
    print(f"Saved {out}")
