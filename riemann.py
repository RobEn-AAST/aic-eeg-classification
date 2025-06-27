# %%
import torch
from modules.competition_dataset import EEGDataset, LABELS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
import mne

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
data_path = './data/mtcaic3'
lda_model_path = './checkpoints/mi/models/lda_mi.pkl'

# Add this at the beginning of your notebook, after imports
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seeds(42)

# %%
window_length = 256
stride = window_length // 3
batch_size = 64

# %%
eeg_channels = [
    "FZ",
    "C3",
    "CZ",
    "C4",
    "PZ",
    "PO7",
    "OZ",
    "PO8",
]

dataset_train = EEGDataset(
    data_path,
    window_length=window_length,
    stride=stride,
    task="mi",
    split="train",
    data_fraction=0.2,
    hardcoded_mean=False,
    eeg_channels=eeg_channels,
)

dataset_val = EEGDataset(
    data_path=data_path,
    window_length=window_length,
    stride=stride,
    task='mi',
    split='validation',
    read_labels=True,
    hardcoded_mean=False,
    eeg_channels=eeg_channels,
)

dataset_test = EEGDataset(
    data_path=data_path,
    window_length=window_length,
    stride=stride,
    task='mi',
    split='test',
    read_labels=False,
    hardcoded_mean=False,
    eeg_channels=eeg_channels,
)

# %%
dataset_train[0][0].shape

# %%
all_data = torch.cat([torch.stack([x for x,_ in ds]) for ds in (dataset_train, dataset_val, dataset_test)])
X_val_train = torch.cat([torch.stack([x for x,_ in ds]) for ds in (dataset_train, dataset_val)])
y_val_train = torch.cat([torch.stack([y for _,y in ds]) for ds in (dataset_train, dataset_val)])

mean = all_data.mean((0, 2))
std = all_data.std((0, 2))

X_val_train = (X_val_train - mean[None, :, None]) / std[None, :, None]

mean, std

# %%
import numpy as np
from sklearn.feature_selection import f_classif

# Concatenate all splits (add dataset_val and dataset_test if needed)
X_all = np.concatenate([
    dataset_train.data.numpy(),
    dataset_val.data.numpy(),
    dataset_test.data.numpy(),
], axis=0)  # shape: [N_total, C, ...]
y_all = np.concatenate([
    dataset_train.labels.numpy(),
    dataset_val.labels.numpy(),
    dataset_test.labels.numpy(),
], axis=0)  # shape: [N_total]

# Detect shape and adapt
if X_all.ndim == 3:
    # [B, C, T]
    num_samples, num_channels, time_points = X_all.shape
    channel_f_scores = []
    for i in range(num_channels):
        channel_data = X_all[:, i, :]  # [N_total, T]
        f_scores_per_timepoint, _ = f_classif(channel_data, y_all)
        aggregated_f_score = np.sum(f_scores_per_timepoint)
        channel_f_scores.append(aggregated_f_score)
elif X_all.ndim == 4:
    # [B, C, F, T]
    num_samples, num_channels, freq_points, time_points = X_all.shape
    channel_f_scores = []
    for i in range(num_channels):
        # Average over freq and time for each channel
        channel_data = X_all[:, i, :, :].mean(axis=(1, 2))  # [N_total]
        f_score, _ = f_classif(channel_data.reshape(-1, 1), y_all)
        channel_f_scores.append(f_score[0])
else:
    raise ValueError(f"Unsupported data shape: {X_all.shape}")

# Optionally, map to channel names
original_channel_names = eeg_channels
channel_scores_dict = {original_channel_names[i]: channel_f_scores[i] for i in range(num_channels)}

print("\n--- F-scores for each channel (higher score indicates more informativeness) ---")
sorted_channels = sorted(channel_scores_dict.items(), key=lambda item: item[1], reverse=True)
for channel, score in sorted_channels:
    print(f"  {channel}: {score:.2f}")

top_3_channels = [channel for channel, score in sorted_channels[:3]]
print(f"\n--- Recommended Top 3 Channels based on F-score: {top_3_channels} ---")

# %%
classifier = "logistic"

clf = None
if classifier == 'logistic':
    clf = LogisticRegression(random_state=42)
elif classifier == 'svm':
    clf = SVC(random_state=42)
elif classifier == 'mdm':
    # Minimum Distance to Mean (Riemannian classifier)
    clf = MDM()

pipeline = Pipeline([
    ('cov', Covariances(estimator='lwf')),  # Ledoit-Wolf shrinkage
    ('tangent', TangentSpace(metric='riemann')),
    ('clf', clf)
])


# Example for train/val/test
X_train = np.stack([x.numpy() for x, y in dataset_train])  # shape: [N, C, T]
y_train = np.array([y[0].item() if hasattr(y, 'item') else y for x, y in dataset_train])

X_val = np.stack([x.numpy() for x, y in dataset_val])
y_val = np.array([y[0].item() if hasattr(y, 'item') else y for x, y in dataset_val])

X_test = np.stack([x.numpy() for x, y in dataset_test])
y_test = np.array([y[0].item() if hasattr(y, 'item') else y for x, y in dataset_test])

# Debug: Check the shapes and types
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train type: {type(y_train[0])}")
print(f"First few labels: {y_train[:5]}")

pipeline.fit(X_train, y_train)