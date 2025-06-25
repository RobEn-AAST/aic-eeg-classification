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
from sklearn.preprocessing import FunctionTransformer
import pywt

def car_func(x):
    return x - np.mean(x, axis=1, keepdims=True)

CarTransformer = FunctionTransformer(
    func = car_func, # x shape: BxCxT
    validate=False,
)

LABELS = ["Left", "Right", "Backward", "Forward"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {i: lbl for i, lbl in enumerate(LABELS)}

_SFREQ = 250
_B, _A = butter(4, [3 / _SFREQ * 2, 100 / _SFREQ * 2], btype="bandpass")  # type: ignore

# encode/decode for unlabeled mode
def position_encode(subj, sess, trial, n_channels=8):
    subj_idx = int(subj[1:]) - 1
    sess_idx = int(sess) - 1
    trial_idx = int(trial) - 1
    return (subj_idx * n_channels + sess_idx) * 10 + trial_idx

def position_decode(code, n_channels=8):
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
        eeg_channels=None,
    ):
        super().__init__()
        task = task.upper()
        self.tmin = int(tmin * _SFREQ)
        default_sizes = {"SSVEP": int(2.0 * _SFREQ), "MI": int(4.5 * _SFREQ)}
        self.win_len = win_len or default_sizes.get(task, int(1.0 * _SFREQ))
        self.channels = (
            ["C3", "PZ", "C4", "OZ", "PO7", "PO8", "CZ", "FZ"]
            if eeg_channels is None
            else eeg_channels
        )
        self.n_channels = len(self.channels)

        self.read_labels = read_labels
        self.lda_n_components = lda_n_components
        self.n_csp = n_csp
        self.split = split.lower()

        self.lda_model_path = os.path.join(checkpoints_dir, f"{task}_lda.pkl")
        self.signal_scalar_path = os.path.join(checkpoints_dir, f"{task}_signal_scaler.pkl")
        self.feature_scalar_path = os.path.join(checkpoints_dir, f"{task}_feature_scaler.pkl")
        self.csp_model_path = os.path.join(checkpoints_dir, f"{task}_csp_model_path.pkl")
        self.car_transformer_path = os.path.join(checkpoints_dir, f"{task}_car_transformer_path.pkl")

        # prepare trial list
        if read_labels:
            meta = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
            meta = meta[meta.task.str.upper() == task]
            if data_fraction < 1.0:
                meta = meta.sample(frac=data_fraction, random_state=42)
            meta = meta.reset_index(drop=True)
            trials = [
                (row.subject_id, row.trial_session, row.trial, row.label)
                for row in meta.itertuples()
            ]
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
                trials = list(
                    rng.choice(trials, size=int(len(trials) * data_fraction), replace=False)
                )

        # cache and load
        eeg_cache = {}
        data_list, label_list, subject_list = [], [], []
        self.trial_ids = []

        for subj, sess, trial, label in trials:
            # capture subject index once at load time
            try:
                subject_num = int(subj[1:])  # e.g. 'S3' -> 3
            except:
                subject_num = int(subj)
            subject_list.append(subject_num)

            key = (subj, sess)
            if key not in eeg_cache:
                fp = os.path.join(data_path, task, split, subj, str(sess), "EEGdata.csv")
                raw = pd.read_csv(fp, usecols=self.channels + ["Validation"]).values
                valid = raw[raw[:, -1] == 1, :-1].T.astype(np.float32)
                eeg_cache[key] = filtfilt(_B, _A, valid, axis=1)
            valid = eeg_cache[key]

            # extract window
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

        # stack and transform
        X_np = np.stack(data_list)  # (B, C, T)
        X_np = self._band_pass_filter(X_np)  # Apply bandpass filter to all data at once
        X_np = self.apply_car(X_np)          # Apply CAR
        freqs = np.linspace(8, 32, 40)
        X_np = self.apply_cwt(X_np, _SFREQ, freqs=freqs)  # (B, C, len(freqs), T)
        X_np = self._normalize_signal(X_np, scalar_path=self.signal_scalar_path)
        
        # finalize tensors
        self.data = torch.tensor(X_np, dtype=torch.float32)               # (B, C, F, T)
        self.labels = torch.tensor(label_list, dtype=torch.long)         # (B,)
        self.subjects = torch.tensor(subject_list, dtype=torch.long) - 1     # (B,)
        # now pack into (B,2): first column = label, second = subject ID
        self.classes = torch.stack((self.labels, self.subjects), dim=1)  # (B,2)

        print(
            f"data shape: {self.data.shape}, classes shape: {self.classes.shape}"
        )

    def apply_car(self, X: np.ndarray) -> np.ndarray:
        """
        Common Average Reference: subtract at each time‐point the mean across channels.
        X: (B, C, T)
        returns X_car: same shape
        """
        if self.split == "train":
            car_transformer = CarTransformer.fit(X)
            joblib.dump(car_transformer, self.car_transformer_path)
        else:
            car_transformer = joblib.load(self.car_transformer_path)

        return np.asarray(car_transformer.transform(X))

    def apply_cwt(self, X: np.ndarray, sfreq: int, freqs: np.ndarray) -> np.ndarray:
        """
        Continuous Wavelet Transform per channel.
        X: (B, C, T)
        sfreq: sampling frequency (e.g. 250)
        freqs: array of desired output frequencies (e.g. np.linspace(8,32,40))
        returns: (B, C, len(freqs), T) scalogram magnitudes
        """
        center_freq = pywt.central_frequency("morl")
        scales = center_freq * sfreq / freqs
        B, C, T = X.shape
        tf_images = np.zeros((B, C, len(freqs), T), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                coef, _ = pywt.cwt(X[b, c], scales, "morl", sampling_period=1 / sfreq)
                tf_images[b, c] = np.abs(coef)
        return tf_images

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
        flat = X_raw.transpose(0, 2, 1).reshape(-1, self.n_channels)  # B*T, C
        if self.split == "train":
            scalar = StandardScaler().fit(flat)
            joblib.dump(scalar, scalar_path)
        else:
            scalar = joblib.load(scalar_path)

        X_norm = scalar.transform(flat)  # (B*T, C)
        X_norm = X_norm.reshape(-1, self.win_len, self.n_channels).transpose(0, 2, 1)  # (B, C, T)

        return X_norm

    def _band_pass_filter(self, data: np.ndarray):
        """
        Apply bandpass filter to data. Expects shape (B, C, T) or (C, T).
        """
        from scipy import signal
        return signal.filtfilt(_B, _A, data, axis=-1)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, i):
        # returns (data, [label, subject_id])
        return self.data[i], self.classes[i]


# decode helper
def decode_label(idx, task):
    return IDX_TO_LABEL[int(idx)]


if __name__ == "__main__":
    # SSVEP (2.0 s window @ 250 Hz)
    dataset_ssvep_train = EEGDataset(
        data_path="./data/mtcaic3",
        task="SSVEP",
        split="train",
        data_fraction=0.2,
        tmin=0.5,  # skip first 0.5 s (125 samples)
        win_len=int(2.0 * _SFREQ),  # 2 s window → 500 samples
        read_labels=True,
    )
    # dataset_ssvep_val = EEGDataset(
    #     data_path="./data/mtcaic3",
    #     task="SSVEP",
    #     split="validation",
    #     tmin=0.5,
    #     win_len=int(2.0 * _SFREQ),
    #     read_labels=True,
    # )

    # # MI (4.5 s window @ 250 Hz)
    # dataset_mi_train = EEGDataset(
    #     data_path="./data/mtcaic3",
    #     task="MI",
    #     split="train",
    #     tmin=0.5,  # skip first 0.5 s (125 samples)
    #     win_len=int(4.5 * _SFREQ),  # 4.5 s window → 1125 samples
    #     read_labels=True,
    # )
    # dataset_mi_val = EEGDataset(
    #     data_path="./data/mtcaic3",
    #     task="MI",
    #     split="validation",
    #     tmin=0.5,
    #     win_len=int(4.5 * _SFREQ),
    #     read_labels=True,
    #     hardcoded_mean=True,
    # )
