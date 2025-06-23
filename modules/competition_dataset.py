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
from sklearn.decomposition import PCA, FastICA
import joblib

LABELS = ["Backward", "Forward", "Left", "Right"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Precompute filter once
_SFREQ = 256
_LOW, _HI = 3, 100
_NYQ = _SFREQ / 2.0
_B, _A = signal.butter(4, [_LOW / _NYQ, _HI / _NYQ], btype="bandpass")
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
        pca_model_path=None,
        ica_model_path=None,
        hardcoded_mean=False,
    ):
        super().__init__()
        assert domain in ("time", "freq", "wavelet")
        assert 0.0 < data_fraction <= 1.0, "data_fraction must be between 0.0 and 1.0"

        task = task.upper()
        self.domain = domain
        self.window_length = window_length
        self.stride = stride or window_length
        self.split = split
        self.read_labels = read_labels
        self.data_fraction = data_fraction

        # eeg_channels = ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]
        eeg_channels = ["PO8", "OZ"]
        usecols = eeg_channels + ["Validation"]

        # Cache for CSVs
        file_cache = {}
        windows = []
        labels = []
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

        # Apply data fraction filtering EARLY - before any CSV reading
        if self.data_fraction < 1.0:
            n_samples = int(len(task_df) * self.data_fraction)
            # Shuffle and take subset for better distribution
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
            if read_labels:
                labels.extend([LABEL_TO_IDX[row.label]] * W.shape[0])
            else:
                labels.extend([position_encode(subj, sess, trial)] * W.shape[0])

        print(f"skipped: {skipped_shit}/{total_stuff}")
        data_array = np.vstack(windows).astype(np.float32)
        labels_np = np.array(labels, dtype=np.int64)

        data_array = self._band_pass_filter(data_array)  # this greatly boosted t-sne

        if self.domain == "freq":
            data_array = self._convert_freq(data_array)
        elif self.domain == "wavelet":
            data_array = self._convert_wavelet_fast(data_array)

        if not hardcoded_mean:
            if self.domain == "wavelet":
                print("Calculating new normalization stats for WAVELET data...")
                self.mean, self.std = self._get_normalization_stats_wavelet(data_array)
            else:
                print("Calculating new normalization stats for TIME/FREQ data...")
                self.mean, self.std = self._get_normalization_stats_3d(data_array)

            print(f"mean: {self.mean} \nstd: {self.std}")
        else:
            self.mean = np.array([-1.6177, -1.9345], dtype=np.float32).reshape(1, -1, 1)
            self.std = np.array([1039.8375, 1004.1708], dtype=np.float32).reshape(1, -1, 1)

            print(f"data shape: {data_array.shape}, mean shape: {self.mean.shape}")

        data_array = self._normalize(data_array)

        # Modified PCA/ICA logic
        if pca_model_path is not None and ica_model_path is not None:
            raise ValueError("Cannot specify both pca_model_path and ica_model_path.")
        elif pca_model_path is not None:
            decomposition_method = "pca"
            model_path = pca_model_path
        elif ica_model_path is not None:
            decomposition_method = "ica"
            model_path = ica_model_path
        else:
            decomposition_method = None
            model_path = ""

        if decomposition_method is not None:
            print(f"Using {decomposition_method.upper()}")
            B, C, T = data_array.shape  # (B, C, T)
            data_array = data_array.reshape(B * T, C)  # (B * T, C)
            assert n_components is not None, "n_components must be specified when using decomposition."

            if self.split == "train":
                if decomposition_method == "pca":
                    self.decomposition_model = PCA(n_components=n_components)
                elif decomposition_method == "ica":
                    self.decomposition_model = FastICA(n_components=n_components)
                self.decomposition_model.fit(data_array)
                joblib.dump(self.decomposition_model, model_path)
                print(f"{decomposition_method.upper()} fitted and saved to {model_path}")
            else:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model path {model_path} does not exist. Run training split first.")
                self.decomposition_model = joblib.load(model_path)
                print(f"{decomposition_method.upper()} model loaded from {model_path}")

            data_array = self.decomposition_model.transform(data_array)  # (B*T, n_components)
            data_array = data_array.reshape(B, T, n_components).transpose(0, 2, 1)  # (B, n_components, T)

        self.data = torch.from_numpy(data_array.copy()).to(torch.float32)
        self.labels = torch.from_numpy(labels_np).to(torch.int64)
        self.trial_ids = trial_ids

    def _convert_freq(self, data: np.ndarray):
        data = np.abs(rfft(data, axis=2))
        return np.log1p(data)

    def _convert_wavelet_fast(self, data: np.ndarray):
        B, C, L = data.shape
        data_tensor = torch.from_numpy(data.copy()).to(torch.float32)

        print(f"Window length (L) is {L}. Calculated a safe J = {self.J}, Q = {self.Q}.")
        data_reshaped = data_tensor.reshape(B * C, L)
        scattering = Scattering1D(J=self.J, shape=(L,), Q=self.Q)
        with torch.no_grad():
            coeffs = scattering(data_reshaped)
        _, n_coeffs, n_time_bins = coeffs.shape
        coeffs_reshaped = coeffs.reshape(B, C, n_coeffs, n_time_bins)
        print("Transform complete.")
        return coeffs_reshaped.numpy()

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

    def _get_normalization_stats_wavelet(self, data: np.ndarray):
        mean = data.mean(axis=(0, 3), keepdims=True)
        std = data.std(axis=(0, 3), keepdims=True) + 1e-6
        print(f"New 4D stats calculated. Mean shape: {mean.shape}")
        return mean, std

    def _normalize(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


# ... (rest of the file, including decode_label, etc.)
def decode_label(idx):
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
    start = time.time()

    # IMPORTANT: To get the correct stats for validation, first run with 'train'
    print("--- First, running on training set to calculate stats ---")
    dataset = EEGDataset(data_path=data_path, window_length=window_length, stride=stride, task="ssvep", split="validation", read_labels=True, hardcoded_mean=True, n_components=3)
