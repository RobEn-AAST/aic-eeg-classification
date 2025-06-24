from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib

LABELS = ["Left", "Right", "Backward", "Forward"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {i: lbl for i, lbl in enumerate(LABELS)}

_SFREQ = 250
_B, _A = butter(4, [3 / _SFREQ * 2, 100 / _SFREQ * 2], btype="bandpass")
n_channels = 8

# encode/decode for unlabeled mode


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
    return f"S{subj_idx+1}", str(sess_idx + 1), str(trial_idx + 1)


class EEGDataset(Dataset):
    def __init__(
        self, data_path, task="SSVEP", split="train", tmin=0.5, win_len=None, eeg_channels=None, read_labels=True, data_fraction=1.0, hardcoded_mean=False, n_components=None, lda_model_path=None
    ):
        super().__init__()
        task = task.upper()
        self.tmin = int(tmin * _SFREQ)
        default_sizes = {"SSVEP": int(2.0 * _SFREQ), "MI": int(4.5 * _SFREQ)}
        self.win_len = win_len or default_sizes.get(task, int(1.0 * _SFREQ))
        # self.channels = eeg_channels or (["PO8", "OZ", "PZ"] if task == "SSVEP" else ["C3", "PZ", "OZ"])
        self.channels = ["C3", "PZ", "C4", "OZ", "PO7", "PO8", "CZ", "FZ"]

        self.read_labels = read_labels

        # prepare trial list
        if read_labels:
            meta = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
            meta = meta[meta.task.str.upper() == task]
            if data_fraction < 1.0:
                meta = meta.sample(frac=data_fraction, random_state=42)
            meta = meta.reset_index(drop=True)
            trials = [(row.subject_id, row.trial_session, row.trial, row.label) for row in meta.itertuples()]
        else:
            trials = []
            base = os.path.join(data_path, task, split)
            for subj in os.listdir(base):
                subj_dir = os.path.join(base, subj)
                if not os.path.isdir(subj_dir):
                    continue
                for sess in os.listdir(subj_dir):
                    eeg_fp = os.path.join(subj_dir, sess, "EEGdata.csv")
                    if not os.path.exists(eeg_fp):
                        continue
                    for trial in range(1, 11):
                        trials.append((subj, sess, trial, None))
            if data_fraction < 1.0:
                rng = np.random.default_rng(42)
                trials = list(rng.choice(trials, size=int(len(trials) * data_fraction), replace=False))

        # cache and load
        eeg_cache = {}
        data_list, label_list = [], []
        self.trial_ids = []
        for subj, sess, trial, label in trials:
            key = (subj, sess)
            if key not in eeg_cache:
                fp = os.path.join(data_path, task, split, subj, str(sess), "EEGdata.csv")
                raw = pd.read_csv(fp, usecols=self.channels + ["Validation"]).values
                valid = raw[raw[:, -1] == 1, :-1].T.astype(np.float32)
                eeg_cache[key] = filtfilt(_B, _A, valid, axis=1)
            valid = eeg_cache[key]
            # extract
            s, e = self.tmin, self.tmin + self.win_len
            seg = valid[:, s:e]
            if seg.shape[1] < self.win_len:
                pad = np.zeros((valid.shape[0], self.win_len - seg.shape[1]), dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=1)
            data_list.append(seg)
            if read_labels:
                idx = LABEL_TO_IDX[label]
            else:
                code = position_encode(subj, sess, trial)
                idx = code
                self.trial_ids.append(code)
            label_list.append(idx)

        # finalize
        self.data = torch.tensor(np.stack(data_list), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(label_list, dtype=torch.long)
        print(f"data shape: {self.data.shape}, label shape: {self.labels.shape}")

        # LDA
        if lda_model_path:
            assert n_components is not None, "n_components must be specified when using LDA"

            X = self.data.permute(0, 2, 1).reshape(-1, self.data.size(1)).numpy()
            y = np.repeat(self.labels.numpy(), self.win_len)
            if split == "train":
                lda = LDA(n_components=n_components)
                lda.fit(X, y)
                joblib.dump(lda, lda_model_path)
            else:
                lda = joblib.load(lda_model_path)
            Xt = lda.transform(X)
            B = self.data.size(0)
            self.data = torch.tensor(Xt.reshape(B, self.win_len, n_components).transpose(0, 2, 1), dtype=torch.float32)

        # normalization
        if hardcoded_mean:
            # m = self.data.mean(dim=(0,2), keepdim=True)
            # s = self.data.std(dim=(0,2), keepdim=True)+1e-6
            if task == "SSVEP":
                m = torch.tensor([-0.0784, 0.0253, -0.2357, -0.1691, 0.0655, -0.0329, -0.1574, -0.1499])
                s = torch.tensor([1801.1117, 1637.9489, 2370.7051, 1784.5072, 2179.6487, 1232.5450, 1088.8466, 1112.8871])
            elif task == "MI":
                m = torch.tensor([0.1557, 0.2455, 0.3131, 0.3464, 0.2734, -0.0244, 0.0947, 0.0542])
                s = torch.tensor([2588.6677, 1692.9707, 3939.0535, 1979.3948, 2239.5142, 1133.0498, 1417.5095, 1076.2388])
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            m = m.view(1, 1, -1, 1)
            s = s.view(1, 1, -1, 1)
            self.data = (self.data - m) / s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]


# decode helper
def decode_label(idx, task):
    return IDX_TO_LABEL[int(idx)]


if __name__ == "__main__":
    # SSVEP (2.0 s window @ 250 Hz)
    dataset_ssvep_train = EEGDataset(
        data_path="./data/mtcaic3",
        task="SSVEP",
        split="train",
        tmin=0.5,  # skip first 0.5 s (125 samples)
        win_len=int(2.0 * _SFREQ),  # 2 s window → 500 samples
        eeg_channels=["PO8", "OZ", "PZ"],
        read_labels=True,
        hardcoded_mean=True,
    )
    dataset_ssvep_val = EEGDataset(
        data_path="./data/mtcaic3",
        task="SSVEP",
        split="validation",
        tmin=0.5,
        win_len=int(2.0 * _SFREQ),
        eeg_channels=["PO8", "OZ", "PZ"],
        read_labels=True,
        hardcoded_mean=True,
    )

    # MI (4.5 s window @ 250 Hz)
    dataset_mi_train = EEGDataset(
        data_path="./data/mtcaic3",
        task="MI",
        split="train",
        tmin=0.5,  # skip first 0.5 s (125 samples)
        win_len=int(4.5 * _SFREQ),  # 4.5 s window → 1125 samples
        eeg_channels=["C3", "PZ", "OZ"],
        read_labels=True,
        hardcoded_mean=True,
    )
    dataset_mi_val = EEGDataset(
        data_path="./data/mtcaic3",
        task="MI",
        split="validation",
        tmin=0.5,
        win_len=int(4.5 * _SFREQ),
        eeg_channels=["C3", "PZ", "OZ"],
        read_labels=True,
        hardcoded_mean=True,
    )
