import numpy as np
import optuna
from sklearn.feature_selection import f_classif
from modules.competition_dataset import EEGDataset  # replace with actual import
import warnings

warnings.filterwarnings("ignore")

# --- User‐configurable ---
data_path = "./data/mtcaic3"            # <-- your data path
original_channel_names = ["FZ","C3","CZ","C4","PZ","PO7","OZ","PO8"]
task = "mi"
data_fraction = 0.7
# -------------------------

def compute_top3_score(window_length, stride, tmin):
    # build datasets
    ds_train = EEGDataset(
        data_path,
        window_length=window_length,
        stride=stride,
        tmin=tmin,
        task=task,
        split="train",
        data_fraction=data_fraction,
        eeg_channels=original_channel_names,
    )
    ds_val = EEGDataset(
        data_path,
        window_length=window_length,
        stride=stride,
        task=task,
        split="validation",
        read_labels=True,
        eeg_channels=original_channel_names,
        tmin=tmin,
    )

    # load into numpy arrays
    X_train = np.stack([x.numpy() for x, _ in ds_train])
    y_train = np.array([y[0].item() for _, y in ds_train])
    X_val = np.stack([x.numpy() for x, _ in ds_val])
    y_val = np.array([y[0].item() for _, y in ds_val])

    # for each channel, compute sum of F-scores over timepoints
    _, num_channels, _ = X_val.shape
    channel_scores = []
    for ch in range(num_channels):
        data_ch = X_val[:, ch, :]
        f_scores, _ = f_classif(data_ch, y_val)
        channel_scores.append(np.sum(f_scores))

    # sum of top-3
    top3_sum = sum(sorted(channel_scores, reverse=True)[:3])
    return top3_sum

def objective(trial):
    wl = trial.suggest_int("window_length", 100, 1300, step=100)
    st = trial.suggest_int("stride", 50, wl, step=50)
    t0 = trial.suggest_int("tmin", 0, 300, step=50)

    return compute_top3_score(wl, st, t0)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=1800)  # e.g. 30 trials or 30 min

    best_wl = study.best_params["window_length"]
    best_stride = study.best_params["stride"]
    best_tmin = study.best_params["tmin"]
    best_score = study.best_value

    print(f"\n=== Best params found ===")
    print(f"window_length = {best_wl}")
    print(f"stride        = {best_stride}")
    print(f"tmin        = {best_tmin}")
    print(f"Top‑3 F‑score sum = {best_score:.2f}\n")

    # re-compute and print full sorted list for the best combo
    ds_val_best = EEGDataset(
        data_path,
        window_length=best_wl,
        stride=best_stride,
        task=task,
        split="validation",
        read_labels=True,
        eeg_channels=original_channel_names,
    )
    X_val_best = np.stack([x.numpy() for x, _ in ds_val_best])
    y_val_best = np.array([y.item() for _, y in ds_val_best])

    channel_scores = []
    for i, ch_name in enumerate(original_channel_names):
        f_scores, _ = f_classif(X_val_best[:, i, :], y_val_best)
        channel_scores.append((ch_name, np.sum(f_scores)))

    sorted_channels = sorted(channel_scores, key=lambda x: x[1], reverse=True)
    print("--- F‑scores per channel (best combo) ---")
    for ch, sc in sorted_channels:
        print(f"{ch:>3}: {sc:.2f}")

    top3 = [ch for ch, _ in sorted_channels[:3]]
    print(f"\nRecommended top 3 channels: {top3}")
