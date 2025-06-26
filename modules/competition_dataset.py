# %%
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
from scipy.fft import fft, rfft
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view
from kymatio.torch import Scattering1D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib

LABELS = ["Backward", "Forward", "Left", "Right"] # ! FOR SSVEP
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Precompute filter once
_SFREQ = 256
_LOW, _HI = 3, 100
_NYQ = _SFREQ / 2.0
_B, _A = signal.butter(4, [_LOW / _NYQ, _HI / _NYQ], btype="bandpass")  # type: ignore
n_channels = 8


class EEGDataset(Dataset):
    def __init__(
        self,
        data_path,
        window_length=128,
        stride=None,
        domain="time",
        trial_length=1750,
        task="SSVEP",
        split="train",
        read_labels=True,
        data_fraction=1.0,
        n_components=3,
        eeg_channels=["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"],
        lda_model_path=None,
        hardcoded_mean=False,
    ):
        super().__init__()
        assert domain in ("time", "freq")
        assert 0.0 < data_fraction <= 1.0, "data_fraction must be between 0.0 and 1.0"
        print(f"task: {task}, split: {split}, domain: {domain}, data_fraction: {data_fraction}")

        task = task.upper()
        self.domain = domain
        self.window_length = window_length
        self.stride = stride or window_length
        self.split = split
        self.read_labels = read_labels
        self.data_fraction = data_fraction

        usecols = eeg_channels + ["Validation"]

        # Cache for CSVs
        file_cache = {}
        windows = []
        labels = []
        subjects = []
        trial_ids = []

        if read_labels:
            labels_df = pd.read_csv(os.path.join(data_path, f"{split}.csv"), usecols=["subject_id", "trial_session", "trial", "task", "label"])
            task_df = labels_df.query(f"task=='{task}'")
        else:
            task_df = []
            task_dir = os.path.join(data_path, task, split)
            for subj in os.listdir(task_dir):
                subj_dir = os.path.join(task_dir, subj)
                if not os.path.isdir(subj_dir):
                    continue
                for sess in os.listdir(subj_dir):
                    sess_dir = os.path.join(subj_dir, sess)
                    eeg_fp = os.path.join(sess_dir, "EEGdata.csv")
                    if not os.path.exists(eeg_fp):
                        continue
                    for trial in range(1, 11):
                        task_df.append({"subject_id": subj, "trial_session": sess, "trial": trial, "label": None})

        if not isinstance(task_df, pd.DataFrame):
            task_df = pd.DataFrame(task_df)

        if self.data_fraction < 1.0:
            n_samples = int(len(task_df) * self.data_fraction)
            task_df = task_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            print(f"Using {self.data_fraction*100:.1f}% of data: {n_samples}/{len(task_df)} samples")

        skipped_shit = 0
        total_stuff = len(task_df)

        for _, row in task_df.iterrows():
            subj, sess, trial = row["subject_id"], str(row["trial_session"]), int(row["trial"])
            fp = os.path.join(data_path, task, split, subj, sess, "EEGdata.csv")
            if not os.path.exists(fp):
                print(f"Warning: missing {fp}")
                continue
            if fp not in file_cache:
                file_cache[fp] = pd.read_csv(fp, usecols=usecols).values
            arr = file_cache[fp]
            start, end = (trial - 1) * trial_length, trial * trial_length
            trial_data = arr[start:end]
            mask = trial_data[:, -1] == 1
            T = trial_data[mask, :-1].T
            if T.shape[1] < self.window_length:
                skipped_shit += 1
                continue
            all_wins = sliding_window_view(T, self.window_length, axis=1)[:, :: self.stride, :]
            W = all_wins.transpose(1, 0, 2)
            windows.append(W)
            n_wins = W.shape[0]
            if read_labels:
                labels.extend([LABEL_TO_IDX[row.label]] * n_wins)
            else:
                labels.extend([position_encode(subj, sess, trial)] * n_wins)
            # Track subject index (as int, e.g. S3 -> 3, then -1 for 0-based)
            try:
                subject_num = int(subj[1:])
            except:
                subject_num = int(subj)
            subject_num -= 1  # Ensure zero-based
            subjects.extend([subject_num] * n_wins)

        print(f"skipped: {skipped_shit}/{total_stuff}")
        data_array = np.vstack(windows).astype(np.float32)
        labels_np = np.array(labels, dtype=np.int64)
        subjects_np = np.array(subjects, dtype=np.int64)

        data_array = self._band_pass_filter(data_array)  # this greatly boosted t-sne

        if self.domain == "freq":
            data_array = self._convert_freq(data_array)

        if task == "SSVEP":
            self.mean = np.array([-1.0309, -0.4789, -0.6384], dtype=np.float32).reshape(1, -1, 1)
            self.std = np.array([2178.9883, 1022.6290,  977.3783], dtype=np.float32).reshape(1, -1, 1)
        elif task == "MI":
            self.mean = np.array([-2.4363, -1.8794, -5.8781, -1.6775, -5.1054, -1.5866, -2.0616, -0.6325], dtype=np.float32).reshape(1, -1, 1)
            self.std = np.array([2598.5059, 1745.9202, 3957.9285, 2063.0957, 2298.0815, 1139.0936, 1412.2756, 1103.5853], dtype=np.float32).reshape(1, -1, 1)
        else:
            raise ValueError(f"Unknown task {task}")

        if hardcoded_mean:
            data_array = self._normalize(data_array)
            print(f"data shape: {data_array.shape}, mean shape: {self.mean.shape}")
        else:
            print("not normalizing...")

        if lda_model_path is not None:
            print(f"Using LDA")
            B, C, T = data_array.shape
            X_all = data_array.transpose(0, 2, 1).reshape(B * T, C)
            y_all = np.repeat(labels_np, T)
            assert n_components is not None, "n_components must be specified when using decomposition."

            if self.split == "train":
                lda = LDA(n_components=n_components)
                lda.fit(X_all, y_all)
                joblib.dump(lda, lda_model_path)
                print(f"LDA fitted and saved to {lda_model_path}")
            else:
                if not os.path.exists(lda_model_path):
                    raise FileNotFoundError(f"Model path {lda_model_path} does not exist. Run training split first.")
                lda = joblib.load(lda_model_path)
                print(f"LDA model loaded from {lda_model_path}")

            data_array = lda.transform(X_all)  # (B*T, n_components)
            data_array = data_array.reshape(B, T, n_components).transpose(0, 2, 1)  # (B, n_components, T)

        self.data = torch.from_numpy(data_array.copy()).to(torch.float32)
        # ...after any label/subject postprocessing...
        self.labels = torch.from_numpy(labels_np).to(torch.int64)
        self.subjects = torch.from_numpy(subjects_np).to(torch.int64)
        if task == "MI" and self.read_labels:
            self.labels -= 2
        self.trial_ids = trial_ids
        # Combine labels and subjects into (B, 2)
        self.classes = torch.stack((self.labels, self.subjects), dim=1)

    def _convert_freq(self, data: np.ndarray):
        data = np.abs(rfft(data, axis=2))  # type:ignore
        return np.log1p(data)

    def _avg_refrencing(self, data: np.ndarray):
        return data - data.mean(axis=2, keepdims=True)

    def _band_pass_filter(self, data: np.ndarray):
        return signal.filtfilt(_B, _A, data, axis=2)

    # --- MISSING HELPER FUNCTIONS ADDED BACK ---
    def _get_normalization_stats_3d(self, data: np.ndarray):
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True) + 1e-6
        print(f"New 3D stats calculated. Mean shape: {mean.shape}")
        return mean, std

    def _normalize(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def __getitem__(self, idx):
        return self.data[idx], self.classes[idx]

    def __len__(self):
        return len(self.data)


# ... (rest of the file, including decode_label, etc.)
def decode_label(idx, task):
    if task.upper() == "MI":
        idx += 2
    return IDX_TO_LABEL[int(idx)]


def encode_label(label):
    return LABEL_TO_IDX[label]


def position_encode(subj, sess, trial):
    subj_idx = int(subj[1:]) - 1
    sess_idx = int(sess) - 1
    trial_idx = int(trial) - 1
    return (subj_idx * n_channels + sess_idx) * 10 + trial_idx


def position_decode(code):
    trial_idx = code % 10
    code //= 10
    sess_idx = code % n_channels
    subj_idx = code // n_channels
    return f"S{subj_idx + 1}", str(sess_idx + 1), str(trial_idx + 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    window_length = 256
    stride = window_length // 3
    data_path = "./data/mtcaic3"
    lda_model_path = "./checkpoints/mi/models/lda_mi.pkl"
    start = time.time()

    dataset = EEGDataset(
        data_path=data_path,
        window_length=window_length,
        stride=stride,
        task="mi",
        split="train",
        data_fraction=0.3,
        read_labels=True,
        hardcoded_mean=False,
        n_components=1,
        lda_model_path=lda_model_path,
    )
    print("done..")
