import os
import torch
import numpy as np
import pandas as pd

# your modules
from modules.competition_dataset import decode_label, position_decode
from modules.competition_dataset import EEGDataset
from modules.moabb_dataset     import load_combined_moabb_data, CompetitionDataset

# braindecode
from braindecode.models import EEGSimpleConv, EEGInceptionERP
from braindecode         import EEGClassifier
from Models              import FilterBankRTSClassifier  # for SSVEP fallback

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths & splits
DATA_PATH   = "./data/mtcaic3"
SPLIT       = "test"

# -----------------------------------------------------------------------------
# Motor Imagery (MOABB) + pretrained EEGSimpleConv / EEGClassifier
# -----------------------------------------------------------------------------
MI_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz"]
MI_TMIN, MI_TMAX = 1.0, 7.0
MI_RESAMPLE = 250

# pretrained MI model hyperparameters
n_chans, n_outputs = len(MI_CHANNELS), 2
n_convs, kernel_size, feature_maps = 2, 8, 112
resampling_freq = 100
activation      = torch.nn.ELU

# build & initialize MI classifier
test_model = EEGSimpleConv(
    n_chans=n_chans,
    n_outputs=n_outputs,
    sfreq=250,
    feature_maps=feature_maps,
    n_convs=n_convs,
    kernel_size=kernel_size,
    resampling_freq=resampling_freq,
    activation=activation,
)
mi_clf = EEGClassifier(
    test_model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=2.13537735805541e-05,
    optimizer__weight_decay=0.0002624302515969059,
    batch_size=128,
    device=device,
    verbose=0,
)
mi_clf.initialize()
mi_clf.module_.load_state_dict(torch.load('./checkpoints/mi/eegsimpleconv.pth', map_location=device))


def run_mi_task():
    # load test data via MOABB
    ds_test = CompetitionDataset(split=SPLIT)
    X_test, _, _, _ = load_combined_moabb_data(
        datasets=[ds_test],
        paradigm_config={
            "channels": MI_CHANNELS,
            "tmin": MI_TMIN, "tmax": MI_TMAX,
            "resample": MI_RESAMPLE,
        }
    )
    probs = mi_clf.predict_proba(X_test)  # shape (50,2)
    # assign IDs 4901..4950
    results = []
    for i in range(probs.shape[0]):
        pred = int(probs[i].argmax())
        results.append({"id": 4901 + i, "label": decode_label(pred, "MI")})
    return results

# -----------------------------------------------------------------------------
# SSVEP pipeline with pretrained EEGInceptionERP + confidence-weighted ensembling
# -----------------------------------------------------------------------------
SS_CHANNELS          = ["OZ", "PO7", "PO8", "PZ"]
SS_WINDOW_LENGTH     = 1000
SS_STRIDE            = 20
SS_MODEL_PATH        = './checkpoints/ssvep/eeginception.pth'

# build & initialize SSVEP classifier (EEGInceptionERP)
ss_model = EEGInceptionERP(
    n_chans=len(SS_CHANNELS),
    n_outputs=4,
    n_times=SS_WINDOW_LENGTH,
    sfreq=250,
    drop_prob=0.5,
    n_filters=8,
    activation=torch.nn.ELU,
)
ss_clf = EEGClassifier(
    ss_model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=2.13537735805541e-05,
    optimizer__weight_decay=0.0002624302515969059,
    batch_size=128,
    device=device,
    verbose=0,
)
ss_clf.initialize()
ss_clf.module_.load_state_dict(torch.load(SS_MODEL_PATH, map_location=device))

def run_ssvep_task():
    # build lookup
    df_meta = pd.read_csv(f"{DATA_PATH}/{SPLIT}.csv")
    lookup = {
        (row.task.upper(), str(row.subject_id), str(row.trial_session), str(row.trial)): row.id
        for row in df_meta.itertuples()
        if row.task.upper() == "SSVEP"
    }
    # load all windows for test
    ds_inf = EEGDataset(
        DATA_PATH, SS_WINDOW_LENGTH, SS_STRIDE,
        task="SSVEP", split=SPLIT, read_labels=False,
        data_fraction=1, tmin=1, eeg_channels=SS_CHANNELS
    )
    windows = [x.numpy() for x,_ in ds_inf]
    codes   = ds_inf.labels.numpy()
    X_windows = np.stack(windows)

    # inference per window
    probs = ss_clf.predict_proba(X_windows)  # shape (N_windows,4)

    # confidence-weighted ensembling per trial
    # weight each window by its max predicted probability (confidence)
    confidences = np.max(probs, axis=1)            # shape (N_windows,)
    weighted   = probs * confidences[:, None]      # broadcast multiply

    results = []
    for code in sorted(np.unique(codes)):
        idxs = np.where(codes == code)[0]
        # weighted average of class probabilities
        agg = weighted[idxs].sum(axis=0) / confidences[idxs].sum()
        pred = int(np.argmax(agg))
        subj, sess, trial = position_decode(int(code))
        key = ("SSVEP", subj, sess, trial)
        if key not in lookup:
            raise KeyError(f"No submission ID for {key}")
        results.append({"id": lookup[key], "label": decode_label(pred, "SSVEP")})
    return results

# -----------------------------------------------------------------------------
# Main: combine & save submission.csv
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mi_res    = run_mi_task()
    ssvep_res = run_ssvep_task()
    pd.DataFrame(mi_res + ssvep_res).sort_values("id").to_csv("submission.csv", index=False)
    print("Saved submission.csv")
