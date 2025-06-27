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
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


def car_func(x):
    return x - np.mean(x, axis=1, keepdims=True)


CarTransformer = FunctionTransformer(
    func=car_func,  # x shape: BxCxT
    validate=False,
)

LABELS = ["Left", "Right", "Backward", "Forward"]  # ! FOR SSVEP
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Precompute filter once
_SFREQ = 250
_NYQ = _SFREQ / 2.0
_LOW, _HI = 3, 30.4
order = 4
_B, _A = signal.butter(order, [_LOW / _NYQ, _HI / _NYQ], btype="bandpass")  # type: ignore
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
        eeg_channels=["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"],
        hardcoded_mean=False,
        checkpoints_dir="./checkpoints",
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

        self.lda_model_path = os.path.join(checkpoints_dir, f"{task}_lda.pkl")
        self.signal_scalar_path = os.path.join(checkpoints_dir, f"{task}_signal_scaler.pkl")
        self.csp_model_path = os.path.join(checkpoints_dir, f"{task}_csp_model_path.pkl")
        self.car_transformer_path = os.path.join(checkpoints_dir, f"{task}_car_transformer_path.pkl")

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
            T = trial_data[mask, :-1][175:].T # skip first 175 points
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

        # preprocessing
        # data_array = self._band_pass_filter(data_array)
        data_array = self._normalize_signal(data_array, scalar_path=self.signal_scalar_path)

        # # normalize bad way
        # if task == "SSVEP":
        #     self.mean = np.array([-1.0309, -0.4789, -0.6384], dtype=np.float32).reshape(1, -1, 1)
        #     self.std = np.array([2178.9883, 1022.6290, 977.3783], dtype=np.float32).reshape(1, -1, 1)
        # elif task == "MI":
        #     self.mean = np.array([-2.4363, -1.8794, -5.8781, -1.6775, -5.1054, -1.5866, -2.0616, -0.6325], dtype=np.float32).reshape(1, -1, 1)
        #     self.std = np.array([2598.5059, 1745.9202, 3957.9285, 2063.0957, 2298.0815, 1139.0936, 1412.2756, 1103.5853], dtype=np.float32).reshape(1, -1, 1)
        # else:
        #     raise ValueError(f"Unknown task {task}")

        # if hardcoded_mean:
        #     data_array = self._normalize(data_array)
        #     print(f"data shape: {data_array.shape}, mean shape: {self.mean.shape}")
        # else:
        #     print("not normalizing...")

        self.data = torch.from_numpy(data_array.copy()).to(torch.float32)
        # ...after any label/subject postprocessing...
        self.labels = torch.from_numpy(labels_np).to(torch.int64)
        self.subjects = torch.from_numpy(subjects_np).to(torch.int64)
        self.trial_ids = trial_ids
        # Combine labels and subjects into (B, 2)
        self.classes = torch.stack((self.labels, self.subjects), dim=1)

    def _normalize_signal(self, X_raw: np.ndarray, scalar_path: str):
        """
        Normalize signal for both (B, C, T) and (B, C, F, T) shapes.
        For (B, C, T): normalize across (B*T, C).
        For (B, C, F, T): normalize across (B*F*T, C).
        """
        if X_raw.ndim == 3:
            # (B, C, T)
            B, C, T = X_raw.shape
            flat = X_raw.transpose(0, 2, 1).reshape(-1, C)  # (B*T, C)
            if self.split == "train":
                scalar = StandardScaler().fit(flat)
                joblib.dump(scalar, scalar_path)
            else:
                scalar = joblib.load(scalar_path)
            X_norm = scalar.transform(flat)
            X_norm = X_norm.reshape(B, T, C).transpose(0, 2, 1)  # (B, C, T)
            return X_norm
        elif X_raw.ndim == 4:
            # (B, C, F, T)
            B, C, F, T = X_raw.shape
            flat = X_raw.transpose(0, 2, 3, 1).reshape(-1, C)  # (B*F*T, C)
            if self.split == "train":
                scalar = StandardScaler().fit(flat)
                joblib.dump(scalar, scalar_path)
            else:
                scalar = joblib.load(scalar_path)
            X_norm = scalar.transform(flat)
            X_norm = X_norm.reshape(B, F, T, C).transpose(0, 3, 1, 2)  # (B, C, F, T)
            return X_norm
        else:
            raise ValueError(f"Unsupported input shape for normalization: {X_raw.shape}")


    def apply_car(self, X: np.ndarray) -> np.ndarray:
        """
        Common Average Reference: subtract at each time‐point the mean across channels.
        X: (B, C, T)
        returns X_car: same shape
        """
        raise ValueError("PROVED TO BE BAD, LOWERING F SCORE AND VALIDATION ACCURACY")
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

    def _convert_freq(self, data: np.ndarray):
        data = np.abs(rfft(data, axis=2))  # type:ignore
        return np.log1p(data)

    def _band_pass_filter(self, data: np.ndarray):
        return signal.filtfilt(_B, _A, data, axis=2)

    def __getitem__(self, idx):
        # fetch raw sample
        x   = self.data[idx]      # shape: (C, T) or (C, F, T)
        cls = self.classes[idx]   # unchanged label/domain tuple

        if self.split == "train":
            # --- 1) time shift (both domains) ---
            shift = np.random.randint(-10, 10)
            x = torch.roll(x, shifts=shift, dims=-1)

            # Only for TF images:
            if x.ndim == 3:  # (C, F, T)
                C, F, T = x.shape

                # --- 2) multiple freq-masks (SpecAugment) ---
                for _ in range(np.random.randint(1, 4)):         # 1–3 masks
                    w  = np.random.randint(1, int(0.15 * F) + 1) # up to 15% of bands
                    f0 = np.random.randint(0, F - w)
                    x[:, f0:f0 + w, :] = 0

                # --- 3) multiple time-masks (SpecAugment) ---
                for _ in range(np.random.randint(1, 4)):         # 1–3 masks
                    w  = np.random.randint(1, int(0.15 * T) + 1) # up to 15% of frames
                    t0 = np.random.randint(0, T - w)
                    x[:, :, t0:t0 + w] = 0

                # --- 4) random frequency shift ---
                shift_f = np.random.randint(-2, 3)  # shift by ±2 bands
                x = torch.roll(x, shifts=shift_f, dims=1)

                # --- 5) amplitude scaling per band ---
                scales = torch.empty(F).uniform_(0.8, 1.2).to(x.device)
                x = x * scales.view(1, F, 1)

            # --- 6) channel dropout (both domains) ---
            if np.random.rand() < 0.1:
                ch = np.random.randint(0, x.size(0))
                x[ch, ...] = 0

            # --- 7) additive Gaussian noise (both domains) ---
            if np.random.rand() < 0.5:
                noise = torch.randn_like(x) * 0.01  # σ = 0.01
                x = x + noise

            # --- 8) mixup on inputs only (both domains) ---
            if np.random.rand() < 0.3:
                j  = np.random.randint(len(self))
                x2 = self.data[j]
                lamda  = np.random.uniform(0.9, 1)
                x  = lamda * x + (1 - lamda) * x2
                # cls is left unchanged

        return x, cls

    def __len__(self):
        return len(self.data)



# ... (rest of the file, including decode_label, etc.)
def decode_label(idx, task):
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
