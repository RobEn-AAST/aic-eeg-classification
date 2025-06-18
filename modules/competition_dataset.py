# %%
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
from scipy.fft import fft
from scipy import signal


LABELS = ['Backward', 'Forward', 'Left', 'Right']
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

class EEGDataset(Dataset):
    def __init__(self, data_path, window_length=128, stride=None, domain="time", trial_length=1750) -> None:
        """
        Args:
            data_path: Path to the SSVEP dataset
            window_length: Length of each window
            stride: Step size between windows
            domain: 'time' or 'freq' - which domain to represent the data in
        """
        super().__init__()
        assert domain in ["time", "freq"], "domain must be either 'time' or 'freq'"
        self.domain = domain
        self.window_length = window_length

        assert trial_length % window_length == 0, "window length must divide by 17500 (17500 is the sampling frequency in Hz)"

        # Load labels first
        labels_df = pd.read_csv(os.path.join(data_path, "train.csv"))
        ssvep_df = labels_df[labels_df["task"] == "SSVEP"]

        self.data = []
        self.labels = []

        eeg_channels = ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]

        # Group by subject/session to avoid reloading the same file
        for _, row in ssvep_df.iterrows():
            subj = row["subject_id"]
            session = str(row["trial_session"])
            trial = int(row["trial"])
            file_path = os.path.join(
                data_path,
                "SSVEP/train",
                subj,
                session,
                "EEGdata.csv",
            )

            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            eeg_data = pd.read_csv(file_path)

            # Get the slice for this trial
            start = (trial - 1) * trial_length
            end = trial * trial_length
            trial_df = eeg_data.iloc[start:end]

            # Filter valid data points using the Validation column
            valid_data = trial_df[trial_df["Validation"] == 1]

            # Select only EEG channels
            trial_data = valid_data[eeg_channels].values.T  # Shape: [C x T]

            # Windowing with stride
            if stride is None:
                stride = window_length

            n_samples = trial_data.shape[1]
            for start_idx in range(0, n_samples - window_length + 1, stride):
                end_idx = start_idx + window_length
                window = trial_data[:, start_idx:end_idx]
                if window.shape[1] == window_length:
                    self.data.append(window.astype(np.float32))
                    self.labels.append(row["label"])

        self.data = np.array(self.data)  # [B x C x T]
        self.labels = np.array([LABEL_TO_IDX[label] for label in self.labels])

        # Data preprocessing
        self.data = self._avg_refrencing(self.data)
        self.data = self._band_pass_filter(self.data)
        if self.domain == "freq":
            self.data = self._convert_freq(self.data)
        self.data = self._normalize(self.data)

        self.data = torch.tensor(self.data) # B x C x T
        self.labels = torch.tensor(self.labels) # [B]

    def _convert_freq(self, data: np.ndarray):
        self.data = np.abs(fft(data, axis=1))
        data = data[
            :,
            :,
            : window_length // 2,
        ]
        data = np.log1p(data)  # log1p = log(1+x) to handle zeros
        print("done converting to freq domain")
        return data

    def _avg_refrencing(self, data: np.ndarray):
        mean = data.mean()
        data -= mean
        return data

    def _band_pass_filter(self, data: np.ndarray):
        # Band pass filtering
        sfreq = 256  # Sampling frequency in Hz
        low_freq = 3  # Lower cutoff frequency
        high_freq = 100  # Upper cutoff frequency
        nyq = sfreq / 2.0  # Nyquist frequency

        # Design the filter
        b, a = signal.butter(
            4, [low_freq / nyq, high_freq / nyq], btype="bandpass"
        )  # Filter order  # Cut-off frequencies  # Filter type

        # Apply the filter to each channel
        for i in range(data.shape[1]):  # Iterate over channels
            data[:, i, :] = signal.filtfilt(b, a, data[:, i, :], axis=1)

        return data

    def _normalize(self, data: np.ndarray):
        # Normalize to [-1, 1]
        data_min = data.min()
        data_max = data.max()
        data = 2 * (data - data_min) / (data_max - data_min) - 1

        return data

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

    def get_freq_mapping(self):
        return {v: k for k, v in self.freq_to_idx.items()}  # idx -> freq mapping

def decode_label(idx):
    return IDX_TO_LABEL[int(idx)]

def encode_label(label):
    return LABEL_TO_IDX[label]

if __name__ == "__main__":
    # %%
    import matplotlib.pyplot as plt

    def get_closest_divisor(target: int):
        target = 160
        n = 17500

        divisors = [i for i in range(1, n+1) if n % i == 0]
        closest = min(divisors, key=lambda x: abs(x - target))
        print(f"Closest divisor of {n} to {target} is {closest}")

    TRIAL_LENGTH = 640
    window_length = 175
    stride = window_length // 3

    dataset_processed = EEGDataset(data_path='/home/zeyadcode/Workspace/ai_projects/eeg_detection/data/mtcaic3', window_length=window_length, stride=stride)