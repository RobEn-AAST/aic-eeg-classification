# %%
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
from scipy.fft import fft, rfft
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
        task="SSVEP",
        split="train",
        read_labels=True,
    ):
        super().__init__()
        assert domain in ("time", "freq")
        # assert trial_length % window_length == 0, f"window length {window_length} doesn't divide trial length {trial_length}"

        self.domain        = domain
        self.window_length = window_length
        self.stride        = stride or window_length
        self.task          = task
        self.split         = split
        self.read_labels   = read_labels

        eeg_channels = ["FZ","C3","CZ","C4","PZ","PO7","OZ","PO8"]
        usecols      = eeg_channels + ["Validation"]

        # Cache for CSVs
        file_cache = {}
        windows = []
        labels  = []
        trial_ids = []

        if read_labels:
            labels_df = pd.read_csv(
                os.path.join(data_path, f"{split}.csv"),
                usecols=["subject_id","trial_session","trial","task","label"]
            )
            task_df = labels_df.query(f"task=='{task}'")
        else:
            # For unlabeled data (validation/test), enumerate all possible files
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
                    # Assume 10 trials per session
                    for trial in range(1, 11):
                        task_df.append({
                            "subject_id": subj,
                            "trial_session": sess,
                            "trial": trial,
                            "label": None  # No label
                        })

        # If not a DataFrame, convert to DataFrame for uniformity
        if not isinstance(task_df, pd.DataFrame):
            task_df = pd.DataFrame(task_df)

        for _, row in task_df.iterrows():
            subj, sess, trial = row["subject_id"], str(row["trial_session"]), int(row["trial"])
            fp = os.path.join(data_path, task, split, subj, sess, "EEGdata.csv")
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
            mask = trial_data[:, -1] == 1
            T    = trial_data[mask, :-1].T  # drop 'Validation' col

            # sliding windows → shape [C x (n_win) x window_length]
            if T.shape[1] < self.window_length:
                print(f'Skipped, T.shape: {T.shape}, self.window_length: {self.window_length}')
                continue
            all_wins = sliding_window_view(T, self.window_length, axis=1)
            # subsample by stride → [C x n_wins' x L]
            all_wins = all_wins[:, ::self.stride, :]
            W = all_wins.transpose(1,0,2)

            if read_labels:
                windows.append(W)
                labels.extend([LABEL_TO_IDX[row.label]] * W.shape[0])
            else:
                windows.append(W)
                trial_code = position_encode(subj, sess, trial)
                labels.extend([trial_code] * W.shape[0])

        # stack into array [B x C x L]
        data_array = np.vstack(windows).astype(np.float32)
        labels_np  = np.array(labels, dtype=np.int64)

        data_array = self._avg_refrencing(data_array)
        data_array = self._band_pass_filter(data_array)

        # Frequency domain?
        if self.domain == "freq":
            data_array = self._convert_freq(data_array)

        data_array = self._normalize(data_array)
        

        # convert to torch
        self.data   = torch.from_numpy(data_array).to(torch.float32)
        self.labels = torch.from_numpy(labels_np).to(torch.int32)
        self.trial_ids = trial_ids

    def _convert_freq(self, data: np.ndarray):
        # rfft → [B x C x (L//2+1)]
        data = np.abs(rfft(data, axis=2))
        data = np.log1p(data)
        return data

    def _avg_refrencing(self, data: np.ndarray):
        return data - data.mean(axis=2, keepdims=True)

    def _band_pass_filter(self, data: np.ndarray):
        return signal.filtfilt(_B, _A, data, axis=2)

    def _normalize(self, data: np.ndarray):
        mean = data.mean(axis=2, keepdims=True)
        std  = data.std(axis=2, keepdims=True) + 1e-6
        print(f"mean: {mean}, std: {std}")
        data = (data - mean) / std
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

def position_encode(subj, sess, trial):
    subj_idx = int(subj[1:]) - 1  # S1→0
    sess_idx = int(sess) - 1      # 1→0
    trial_idx = int(trial) - 1    # 1→0
    return (subj_idx * 8 + sess_idx) * 10 + trial_idx

def position_decode(code):
    # Reverse of: (subj_idx * 8 + sess_idx) * 10 + trial_idx
    trial_idx = code % 10
    code //= 10
    sess_idx = code % 8
    subj_idx = code // 8
    subj = f"S{subj_idx + 1}"
    sess = str(sess_idx + 1)
    trial = str(trial_idx + 1)
    return subj, sess, trial

if __name__ == "__main__":
# %%

    # %%
    import matplotlib.pyplot as plt
    import time

    TRIAL_LENGTH = 640
    window_length = 175
    stride = window_length // 3
    data_path='./data/mtcaic3'

    start = time.time()
    dataset= EEGDataset(data_path, window_length=window_length, stride=stride, task="SSVEP", split="train", read_labels=True)
    print(f"time taken: {time.time() - start}")
    print("hello")