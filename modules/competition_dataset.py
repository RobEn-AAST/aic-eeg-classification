# %%
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
from scipy.fft import fft
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view


LABELS = ['Backward', 'Forward', 'Left', 'Right']
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Precompute filter once
_SFREQ    = 256
_LOW, _HI = 3, 100
_NYQ      = _SFREQ / 2.0
_B, _A    = signal.butter(4, [_LOW/_NYQ, _HI/_NYQ], btype='bandpass')

class EEGDataset(Dataset):
    def __init__(
        self,
        data_path,
        window_length=128,
        stride=None,
        domain="time",
        trial_length=1750,
    ):
        super().__init__()
        assert domain in ("time", "freq")
        assert trial_length % window_length == 0, f"widnow length {window_length} doesn't divide trial length {trial_length}"
        
        self.domain        = domain
        self.window_length = window_length
        self.stride        = stride or window_length

        labels_df = pd.read_csv(
            os.path.join(data_path, "train.csv"),
            usecols=["subject_id","trial_session","trial","task","label"]
        )
        ssvep     = labels_df.query("task=='SSVEP'")
        
        eeg_channels = ["FZ","C3","CZ","C4","PZ","PO7","OZ","PO8"]
        usecols      = eeg_channels + ["Validation"]

        # Cache for CSVs
        file_cache = {}
        windows = []
        labels  = []

        for _, row in ssvep.iterrows():
            subj, sess, trial = row.subject_id, str(row.trial_session), int(row.trial)
            fp = os.path.join(data_path, "SSVEP","train",subj,sess,"EEGdata.csv")
            if not os.path.exists(fp):
                print(f"Warning: missing {fp}")
                continue

            # load once
            if fp not in file_cache:
                file_cache[fp] = pd.read_csv(fp, usecols=usecols).values
            arr = file_cache[fp]

            start = (trial - 1) * trial_length
            end   = trial     * trial_length
            trial_data = arr[start:end]
            # filter valid rows and transpose → shape [C x T]
            mask = trial_data[:, -1] == 1
            T    = trial_data[mask, :-1].T  # drop 'Validation' col

            # sliding windows → shape [C x (n_win) x window_length]
            if T.shape[1] < self.window_length:
                print(f'Skipped, T.shape: {T.shape}, self.window_length: {self.window_length}')
                continue
            all_wins = sliding_window_view(T, self.window_length, axis=1)
            # subsample by stride → [C x n_wins' x L]
            all_wins = all_wins[:, ::self.stride, :]
            # reshape → [n_wins' x C x L]
            W = all_wins.transpose(1,0,2)
            
            windows.append(W)
            labels.extend([LABEL_TO_IDX[row.label]] * W.shape[0])

        # stack into array [B x C x L]
        data_array = np.vstack(windows).astype(np.float32)
        labels_np  = np.array(labels, dtype=np.int64)

        # Avg reference per channel
        data_array -= data_array.mean(axis=2, keepdims=True)

        # Band‑pass filter all at once over axis=2
        data_array = signal.filtfilt(_B, _A, data_array, axis=2)

        # Frequency domain?
        if self.domain == "freq":
            # rfft → [B x C x (L//2+1)]
            data_array = np.abs(rfft(data_array, axis=2))
            data_array = np.log1p(data_array)

        # Normalize to zero‑mean, unit‑var per channel
        mean = data_array.mean(axis=2, keepdims=True)
        std  = data_array.std(axis=2, keepdims=True) + 1e-6
        data_array = (data_array - mean) / std

        # convert to torch
        self.data   = torch.from_numpy(data_array)
        self.labels = torch.from_numpy(labels_np)


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
    import time

    TRIAL_LENGTH = 640
    window_length = 175
    stride = window_length // 3

    start = time.time()
    dataset_processed = EEGDataset(data_path='./data/mtcaic3', window_length=window_length, stride=stride)
    print(f"time taken: {time.time() - start}")
    print("hello")