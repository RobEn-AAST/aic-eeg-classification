from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
from scipy.fft import fft
from scipy import signal


class EEGDataset(Dataset):
    def __init__(self, data_path, trial_length, window_length=128, stride=None, domain="time") -> None:
        """
        Args:
            data_path: Path to the SSVEP dataset
            trial_length: Number of samples before frequency shift
            window_length: Length of each window
            stride: Step size between windows
            domain: 'time' or 'freq' - which domain to represent the data in
        """
        super().__init__()
        assert domain in ["time", "freq"], "domain must be either 'time' or 'freq'"
        self.domain = domain

        assert trial_length % window_length == 0, "Please choose window_length that divides by trial_length"
        self.data_path = data_path
        self.data = []
        self.labels = []

        if stride == None:
            stride = window_length

        # Load all subjects' data
        subject_dirs = [d for d in os.listdir(data_path) if d.startswith("subject_")]

        for subject_dir in subject_dirs:
            subject_path = os.path.join(data_path, subject_dir)
            sample_files = [f for f in os.listdir(subject_path) if f.endswith(".csv")]

            for sample_file in sample_files:
                sample_file_path = os.path.join(subject_path, sample_file)
                df = pd.read_csv(sample_file_path, header=None, skiprows=1)  # samples x (electrodes + 1)

                freqs = df.iloc[:, -1].values

                # first get of shape trial_length x freq
                n_rows = len(freqs)
                n_trials = n_rows // trial_length
                for t in range(n_trials):
                    start = t * trial_length
                    end = start + trial_length
                    block_freqs = freqs[start:end]  # shape Nx1

                    assert np.all(block_freqs == block_freqs[0]), f"Mixed labels in trial {t} of {sample_file}"

                    trial_label = block_freqs[0]
                    trial_data = df.iloc[start:end, :-1].values  # shape [trial_length x C]

                    for i in range(0, trial_length - window_length + 1, stride):
                        win = trial_data[i : i + window_length, :].T  # C x tiral_length
                        self.data.append(win.astype(np.float32))
                        self.labels.append([trial_label])

        self.data = np.array(self.data)  # B x c x trial_length
        self.labels = np.array(self.labels)  # B x 1 = 5200 x 1

        # Data preprocessing
        self.data = self._avg_refrencing(self.data)
        self.data = self._band_pass_filter(self.data)
        if self.domain == "freq":
            self.data = self._convert_freq(self.data)
        self.data = self._normalize(self.data)

        unique_freqs = np.unique(np.array([label[0] for label in self.labels]))
        self.freq_to_idx = {freq: idx for idx, freq in enumerate(unique_freqs)}
        self.labels = np.array([self.freq_to_idx[label[0]] for label in self.labels])

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

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
        b, a = signal.butter(4, [low_freq / nyq, high_freq / nyq], btype="bandpass")  # Filter order  # Cut-off frequencies  # Filter type

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


if __name__ == "__main__":
    import kagglehub
    import matplotlib.pyplot as plt

    path = kagglehub.dataset_download("girgismicheal/steadystate-visual-evoked-potential-signals")
    path += "/SSVEP (BrainWheel)"
    TRIAL_LENGTH = 640
    window_length = 160
    stride = 3

    # Create two datasets - one with preprocessing, one without
    dataset_processed = EEGDataset(path, TRIAL_LENGTH, window_length, stride=stride)
    
    class RawEEGDataset(EEGDataset):
        def _avg_refrencing(self, data): return data
        def _band_pass_filter(self, data): return data
        def _normalize(self, data): return data
    
    dataset_raw = RawEEGDataset(path, TRIAL_LENGTH, window_length, stride=stride)

    # Get same random sample from both datasets
    idx = np.random.randint(len(dataset_processed))
    raw_data, label = dataset_raw[idx]
    processed_data, _ = dataset_processed[idx]

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot raw data
    time = np.arange(window_length) / 256  # Convert to seconds
    for ch in range(raw_data.shape[0]):
        ax1.plot(time, raw_data[ch].numpy(), label=f'Channel {ch+1}', alpha=0.7)
    ax1.set_title('Raw EEG Signals')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    # Plot processed data
    for ch in range(processed_data.shape[0]):
        ax2.plot(time, processed_data[ch].numpy(), label=f'Channel {ch+1}', alpha=0.7)
    ax2.set_title(f'Processed EEG Signals (Label: {dataset_raw.get_freq_mapping()[label.item()]} Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Normalized Amplitude')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
