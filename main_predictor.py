import torch, numpy as np, pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from modules.competition_dataset import EEGDataset, decode_label, position_decode
from Models import get_mi_model, get_ssvep_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./data/mtcaic3"
batch_size = 64
window_length = 256
stride = window_length // 3
confidence_exponent = 2.0

models = {
    "SSVEP": {
        "builder": get_ssvep_model,
        "ckpt": "./checkpoints/ssvep/models/ssvep_PO8_OZ_PZ.pth",
        "channels": ["PO8", "OZ", "PZ"]
    },
    "MI": {
        "builder": get_mi_model,
        "ckpt": "./checkpoints/mi/models/the_honored_one.pth",
        "channels": ["C3", "PZ", "OZ"]
    }
}

def run_task(task, split="validation"):
    cfg = models[task]
    model = cfg["builder"]().to(device)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
    df = pd.read_csv(f"{data_path}/{split}.csv")
    lookup = {
        (str(r.subject_id), str(r.trial_session), str(r.trial)): r.id
        for r in df.itertuples()
    }

    ds = EEGDataset(
        data_path=data_path,
        window_length=window_length,
        stride=stride,
        domain="time",
        task=task,
        split=split,
        read_labels=False,
        hardcoded_mean=True,
        data_fraction=1,
        eeg_channels=cfg["channels"]
    )
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)

    logits = np.concatenate([
        model(x.to(device)).cpu().numpy()
        for x, _ in tqdm(loader, desc=f"{task} inference")
    ], axis=0)

    trial_codes = ds.labels.numpy()
    results = []
    for code in sorted(set(trial_codes)):
        inds = np.where(trial_codes == code)[0]
        windows = torch.from_numpy(np.array([logits[i] for i in inds])[2:3])
        if windows.numel() == 0:
            continue
        probs = F.softmax(windows, dim=1)
        conf = probs.max(1).values ** confidence_exponent
        if conf.sum() == 0:
            avg = windows.mean(0)
        else:
            avg = (windows * conf.unsqueeze(1)).sum(0) / conf.sum()
        pred = decode_label(avg.numpy().argmax())
        subj, sess, trial = position_decode(int(code))
        csv_id = lookup.get((subj, sess, trial))
        if csv_id is not None:
            results.append({"id": csv_id, "label": pred})

    out = f"submission_{task}.csv"
    pd.DataFrame(results).sort_values("id").to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    for t in models:
        run_task(t)
