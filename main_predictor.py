import torch, numpy as np, pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from modules.competition_dataset import EEGDataset, decode_label, position_decode
from Models import get_mi_model, get_ssvep_model
import joblib
from pyriemann.estimation import Covariances
from scipy.signal import sosfiltfilt, butter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split = "validation"
data_path = "./data/mtcaic3"
batch_size = 64
confidence_exponent = 2.0

# Filter bank for MI
bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
sos_bands = [butter(4, (l / 125, h / 125), btype="bandpass", output="sos") for l, h in bands]

def compute_fb_covs(X):
    """X: (n_trials, C, T) → fb_covs: (n_trials, B, C, C)"""
    n, C, _ = X.shape
    B = len(sos_bands)
    fb_covs = np.zeros((n, B, C, C))
    for i, sos in enumerate(sos_bands):
        Xf = sosfiltfilt(sos, X, axis=2)
        fb_covs[:, i] = Covariances(estimator="lwf").transform(Xf)
    return fb_covs

def predict_fb_rts(ts, w, clf, X):
    """Given fitted ts, weights w, and clf, predict on new X."""
    fb_covs = compute_fb_covs(X)
    n, B, C, _ = fb_covs.shape
    covs_flat = fb_covs.reshape(n * B, C, C)
    Z = ts.transform(covs_flat).reshape(n, B, -1)
    Z_weighted = np.concatenate([np.sqrt(w[i]) * Z[:, i, :] for i in range(B)], axis=1)
    return clf.predict(Z_weighted)

def run_mi_task(lookup, results, split=split):
    # Load saved Riemannian filter-bank model
    ts, w, clf = joblib.load("./checkpoints/mi/fb_rts_fsvm_model.joblib")
    eeg_channels = [
        "FZ",
        "C3",
        "CZ",
        "C4",
        "PO7",
        "PO8",
    ]
    window_length = 1000
    tmin = 250
    stride = window_length // 2
    ds = EEGDataset(
        data_path=data_path,
        window_length=window_length,
        tmin=tmin,
        stride=stride,
        domain="time",
        task="MI",
        split=split,
        read_labels=False,
        data_fraction=1,
        eeg_channels=eeg_channels,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)
    all_X = []
    for x, _ in tqdm(loader, desc="MI inference"):
        all_X.append(x.numpy())
    all_X = np.concatenate(all_X, axis=0)  # [N, C, T]
    preds = predict_fb_rts(ts, w, clf, all_X)
    codes = ds.labels.numpy()
    for code in sorted(set(codes)):
        inds = np.where(codes == code)[0]
        if len(inds) == 0:
            continue
        pred = preds[inds[2]] if len(inds) > 2 else preds[inds[0]]
        label = decode_label(int(pred), "MI")
        subj, sess, trial = position_decode(int(code))
        key = ("MI", subj, sess, trial)
        csv_id = lookup.get(key)
        if csv_id is not None:
            results.append({"id": csv_id, "label": label})
        else:
            raise ValueError(f"{key} IS NOT FOUND: {csv_id}")

def run_ssvep_task(lookup, results, split=split):
    # Load saved SSVEP model (PyTorch neural net)
    eeg_channels = ["PO8", "OZ", "PZ"]
    window_length = 256  # Adjust as needed for your SSVEP model
    tmin = 0
    stride = window_length // 2
    model = get_ssvep_model().to(device)
    model.load_state_dict(torch.load("./checkpoints/ssvep/models/ssvep_PO8_OZ_PZ.pth", map_location=device))
    model.eval()
    ds = EEGDataset(
        data_path=data_path,
        window_length=window_length,
        tmin=tmin,
        stride=stride,
        domain="time",
        task="SSVEP",
        split=split,
        read_labels=False,
        data_fraction=1,
        eeg_channels=eeg_channels,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)
    all_logits = np.concatenate([model(x.to(device)).cpu().detach().numpy() for x, _ in tqdm(loader, desc="SSVEP inference")], axis=0)
    codes = ds.labels.numpy()
    for code in sorted(set(codes)):
        inds = np.where(codes == code)[0]
        wins = torch.from_numpy(np.stack([all_logits[i] for i in inds])[2:3])
        if wins.numel() == 0:
            continue
        probs = F.softmax(wins, dim=1)
        conf = probs.max(1).values ** confidence_exponent
        avg = (wins * conf.unsqueeze(1)).sum(0) / (conf.sum() or 1)
        pred = decode_label(int(avg.argmax()), "SSVEP")
        subj, sess, trial = position_decode(int(code))
        key = ("SSVEP", subj, sess, trial)
        csv_id = lookup.get(key)
        if csv_id is not None:
            results.append({"id": csv_id, "label": pred})
        else:
            raise ValueError(f"{key} IS NOT FOUND: {csv_id}")

if __name__ == "__main__":
    # build a lookup that includes task
    df = pd.read_csv(f"{data_path}/{split}.csv")
    lookup = {(row.task, str(row.subject_id), str(row.trial_session), str(row.trial)): row.id for row in df.itertuples()}

    results = []
    run_mi_task(lookup, results)
    run_ssvep_task(lookup, results)

    # save one combined CSV
    out = "submission.csv"
    pd.DataFrame(results).sort_values("id").to_csv(out, index=False)
    print(f"Saved {out}")