import torch, numpy as np, pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from modules.competition_dataset import EEGDataset, decode_label, position_decode
from Models import get_mi_model, get_ssvep_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split = "validation"
data_path = "./data/mtcaic3"
batch_size = 64
window_length = 256
stride = window_length // 3
confidence_exponent = 2.0

# define your two models
models = {
    "SSVEP": {"builder": get_ssvep_model, "ckpt": "./checkpoints/ssvep/models/ssvep_PO8_OZ_PZ.pth", "channels": ["PO8", "OZ", "PZ"]},
    "MI": {"builder": get_mi_model, "ckpt": "./checkpoints/mi/models/the_honored_one.pth", "channels": ["C3", "PZ", "C4"]},
}


def run_task(task, lookup, results, split=split):
    """Run inference for a single task, append to results."""
    cfg = models[task]
    model = cfg["builder"]().to(device)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
    model.eval()

    # dataset & loader
    ds = EEGDataset(
        data_path=data_path, window_length=window_length, stride=stride, domain="time", task=task, split=split, read_labels=False, hardcoded_mean=True, data_fraction=1, eeg_channels=cfg["channels"]
    )
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)

    # run all windows once
    all_logits = np.concatenate([model(x.to(device)).cpu().detach().numpy() for x, _ in tqdm(loader, desc=f"{task} inference")], axis=0)

    # aggregate per trial code
    codes = ds.labels.numpy()
    for code in sorted(set(codes)):
        inds = np.where(codes == code)[0]
        # take the 3rd window only (as before)
        wins = torch.from_numpy(np.stack([all_logits[i] for i in inds])[2:3])
        if wins.numel() == 0:
            continue

        probs = F.softmax(wins, dim=1)
        conf = probs.max(1).values ** confidence_exponent
        avg = (wins * conf.unsqueeze(1)).sum(0) / (conf.sum() or 1)
        pred = decode_label(int(avg.argmax()), task)

        subj, sess, trial = position_decode(int(code))
        key = (task, subj, sess, trial)
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
    for t in models:
        run_task(t, lookup, results)

    # save one combined CSV
    out = "submission.csv"
    pd.DataFrame(results).sort_values("id").to_csv(out, index=False)
    print(f"Saved {out}")
