import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.competition_dataset import EEGDataset, decode_label, position_decode
from Models.ssvep_classifier import SSVEPClassifier
from Models.ssvep_best_params import ssvep_best_params

def main():
    # --- Config ---
    data_path = './data/mtcaic3'
    split = "validation"
    task = "SSVEP"
    model_path = "./checkpoints/ssvep/models/75_lstm.pth"
    output_csv = "submission.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load validation.csv for trial IDs ---
    val_csv = pd.read_csv(f"{data_path}/{split}.csv")
    # Map (subject_id, trial_session, trial) -> id
    id_lookup = {}
    for _, row in val_csv.iterrows():
        key = (str(row["subject_id"]), str(row["trial_session"]), str(row["trial"]))
        id_lookup[key] = row["id"]

    # --- Load Dataset ---
    window_length = ssvep_best_params["window_length"]
    stride = window_length // 3 # ssvep_best_params["stride_factor"]
    dataset = EEGDataset(
        data_path=data_path,
        window_length=window_length,
        stride=stride,
        task=task,
        split=split,
        read_labels=False,
    )

    # --- Load Model ---
    n_electrodes = dataset.data.shape[1]
    n_samples = dataset.data.shape[2]
    model = SSVEPClassifier(
        n_electrodes=n_electrodes,
        n_samples=n_samples,
        out_dim=4,
        dropout=ssvep_best_params["dropout"],
        kernLength=ssvep_best_params["kernLength"],
        F1=ssvep_best_params["F1"],
        D=ssvep_best_params["D"],
        F2=ssvep_best_params["F2"],
        hidden_dim=ssvep_best_params["hidden_dim"],
        layer_dim=ssvep_best_params["layer_dim"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Run Inference ---
    all_logits = []
    batch_size = ssvep_best_params["batch_size"]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, _ in tqdm(data_loader, desc="Predicting"):
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)  # [N_windows, n_classes]

    # --- Aggregate logits by trial code ---
    trial_codes = dataset.labels.numpy()
    trial_logits = {}
    for idx, code in enumerate(trial_codes):
        if code not in trial_logits:
            trial_logits[code] = []
        trial_logits[code].append(all_logits[idx])

    # --- Average logits and get predictions ---
    results = []
    for code in sorted(trial_logits):
        logits_list = trial_logits[code]
        avg_logits = np.mean(logits_list, axis=0)
        pred_idx = np.argmax(avg_logits)
        pred_label = decode_label(pred_idx)
        subj, sess, trial = position_decode(code)
        # Use the tuple to get the correct id from validation.csv
        csv_id = id_lookup[(subj, sess, trial)]
        results.append({"id": csv_id, "label": pred_label})

    # --- Save to CSV ---
    df = pd.DataFrame(results)
    df = df.sort_values("id")
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

if __name__ == '__main__':
    main()