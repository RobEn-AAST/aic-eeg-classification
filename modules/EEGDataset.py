from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch

class EEGDataset(Dataset):
    def __init__(self, data_path, trial_length, window_length=128, stride=None) -> None:
        """
        todo complete documentation
        trial_length: the number of rows before frequency shift in the dataset

        N: sample length
        C: channels (number of electrodes)
        B: Batch Size
        """
        super().__init__()

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
                        win = trial_data[i: i + window_length, :] # trial_window x C
                        self.data.append(win.astype(np.float32))
                        self.labels.append([trial_label])


        self.data = np.array(self.data)  # B x window_length x c
        self.labels = np.array(self.labels) # B x 1 = 5200 x 1 

        unique_freqs = np.unique(np.array([label[0] for label in self.labels]))
        self.freq_to_idx = {freq: idx for idx, freq in enumerate(unique_freqs)}
        self.labels = np.array([self.freq_to_idx[label[0]] for label in self.labels])

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

    def get_freq_mapping(self):
        return {v: k for k, v in self.freq_to_idx.items()}  # idx -> freq mapping


