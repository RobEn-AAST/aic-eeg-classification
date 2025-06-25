from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler

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
        self,
        data_path,
        task="SSVEP",
        split="train",
        tmin=0.5,
        win_len=None,
        read_labels=True,
        data_fraction=1.0,
        lda_n_components=3,
        n_csp=4,
        checkpoints_dir="./checkpoints/",
    ):
        super().__init__()
        task = task.upper()
        self.tmin = int(tmin * _SFREQ)
        default_sizes = {"SSVEP": int(2.0 * _SFREQ), "MI": int(4.5 * _SFREQ)}
        self.win_len = win_len or default_sizes.get(task, int(1.0 * _SFREQ))
        self.channels = ["C3", "PZ", "C4", "OZ", "PO7", "PO8", "CZ", "FZ"]

        self.read_labels = read_labels
        self.lda_n_components = lda_n_components
        self.n_csp = n_csp
        self.split = split.lower()

        self.lda_model_path = os.path.join(checkpoints_dir, f"{task}_lda.pkl")
        self.signal_scalar_path = os.path.join(checkpoints_dir, f"{task}_signal_scaler.pkl")
        self.feature_scalar_path = os.path.join(checkpoints_dir, f"{task}_feature_scaler.pkl")
        self.csp_model_path = os.path.join(checkpoints_dir, f"{task}_csp_model_path.pkl")

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

        X_np = np.stack(data_list)  # BxCxT
        y_np = np.array(label_list)  # B

        X_np = self._normalize_signal(X_np, self.signal_scalar_path)
        X_np = self.apply_csp(X_np, y_np)
        # X_np = self._normalize_signal(X_np, self.feature_scalar_path)
        # X_np = self.apply_lda(X_np)

        # finalize
        self.data = torch.tensor(X_np, dtype=torch.float32)
        self.labels = torch.tensor(label_list, dtype=torch.long)
        print(f"data shape: {self.data.shape}, label shape: {self.labels.shape}")

    def apply_csp(self, X: np.ndarray, y: np.ndarray):
        if self.split == "train":
            csp = CSP(n_components=self.n_csp, reg="ledoit_wolf", log=None, transform_into="csp_space")
            X_filt = csp.fit_transform(X, y)  # (B, n_csp, T)
            joblib.dump(csp, self.csp_model_path)
        else:
            csp = joblib.load(self.csp_model_path)
            X_filt = csp.transform(X)  # (B, n_csp, T)

        return X_filt

    def _normalize_signal(self, X_raw: np.ndarray, scalar_path: str):
        flat = X_raw.transpose(0, 2, 1).reshape(-1, n_channels)  # B*T, C
        if self.split == "train":
            scalar = StandardScaler().fit(flat)
            joblib.dump(scalar, scalar_path)
        else:
            scalar = joblib.load(scalar_path)

        X_norm = scalar.transform(flat)  # (B*T, C)
        X_norm = X_norm.reshape(-1, self.win_len, n_channels).transpose(0, 2, 1)  # (B, C, T)

        return X_norm

    # def apply_lda(self, data: np.ndarray):
    #     assert self.lda_n_components is not None, "n_components must be specified when using LDA"

    #     X = data.permute(0, 2, 1).reshape(-1, self.data.size(1)).numpy()
    #     y = np.repeat(self.labels.numpy(), self.win_len)
    #     if self.split == "train":
    #         lda = LDA(n_components=self.lda_n_components)
    #         lda.fit(X, y)
    #         joblib.dump(lda, self.lda_model_path)
    #     else:
    #         lda = joblib.load(self.lda_model_path)
    #     Xt = lda.transform(X)
    #     B = data.size(0)
    #     data = torch.tensor(Xt.reshape(B, self.win_len, self.lda_n_components).transpose(0, 2, 1), dtype=torch.float32)

        return data

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
        read_labels=True,
        hardcoded_mean=True,
    )
    dataset_ssvep_val = EEGDataset(
        data_path="./data/mtcaic3",
        task="SSVEP",
        split="validation",
        tmin=0.5,
        win_len=int(2.0 * _SFREQ),
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
        read_labels=True,
        hardcoded_mean=True,
    )
    dataset_mi_val = EEGDataset(
        data_path="./data/mtcaic3",
        task="MI",
        split="validation",
        tmin=0.5,
        win_len=int(4.5 * _SFREQ),
        read_labels=True,
        hardcoded_mean=True,
    )
