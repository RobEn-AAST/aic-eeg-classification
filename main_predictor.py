import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F  # <-- Add this import

from modules.competition_dataset import EEGDataset, decode_label, position_decode
from Models import ssvep_best_params

# It's good practice to define the model class in the script that uses it
# or import it properly. I'm including it here for completeness.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class DepthWiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, dim_mult=1, padding=0, bias=False):
        super(DepthWiseConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * dim_mult, padding=padding, kernel_size=kernel_size, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.depthwise(x)


class SeperableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(SeperableConv2D, self).__init__()
        self.depthwise = DepthWiseConv2D(in_channels, kernel_size, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SSVEPClassifier(nn.Module):
    # EEG Net Based
    # todo look at this https://paperswithcode.com/paper/a-transformer-based-deep-neural-network-model
    def __init__(self, n_electrodes=16, out_dim=4, dropout=0.25, kernLength=256, F1=96, D=1, F2=96, hidden_dim=100, layer_dim=1):
        super().__init__()

        # B x C x T
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            #
            DepthWiseConv2D(F1, (n_electrodes, 1), dim_mult=D, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),  # todo try making this max pool
            nn.Dropout(dropout),
            #
            SeperableConv2D(F1 * D, F2, kernel_size=(1, 16), padding="same", bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )

        self.lstm_head = LSTMModel(F2, hidden_dim, layer_dim, out_dim)

    def forward(self, x: torch.Tensor):
        """expected input shape: BxCxT"""
        x = x.unsqueeze(1)
        y = self.block_1(x)  # B x F1 x 1 x time_sub

        y = y.squeeze(2)  # B x F1 x time_sub
        y = y.permute(0, 2, 1)  # B x time_sub x F1
        y = self.lstm_head(y)

        return y


model = SSVEPClassifier(
    n_electrodes=3,
    dropout=0.33066508963955576,
    kernLength=64,
    F1 = 8,
    D = 2,
    F2 = 32,
    hidden_dim=128,
    layer_dim=1,
).to(device)

batch_size = 64
window_length = 64 * 5
stride = window_length // 2
model_path = "./checkpoints/ssvep/models/pca_ssvep.pth"
pca_model_path = './checkpoints/ssvep/models/pca.pkl'
ica_model_path = './checkpoints/ssvep/models/ica.pkl'

def main():
    # --- Config ---
    data_path = "./data/mtcaic3"
    split = "validation"
    task = "SSVEP"
    output_csv = "submission.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameter for confidence weighting ---
    # A higher value gives more power to very confident predictions.
    # 1.0 is a standard weighted average. 2.0 or 3.0 is more aggressive.
    confidence_exponent = 2.0

    # --- Load validation.csv for trial IDs ---
    val_csv = pd.read_csv(f"{data_path}/{split}.csv")
    id_lookup = {}
    for _, row in val_csv.iterrows():
        key = (str(row["subject_id"]), str(row["trial_session"]), str(row["trial"]))
        id_lookup[key] = row["id"]

    # --- Load Dataset ---
    dataset = EEGDataset(
        data_path=data_path,
        window_length=window_length,
        stride=stride,
        task=task,
        split=split,
        read_labels=False,
        pca_model_path=pca_model_path,
        hardcoded_mean=True,
        n_components=3
    )

    # --- Load Model ---
    # Using the ssvep_best_params from your notebook for model instantiation

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Run Inference ---
    all_logits = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, _ in tqdm(data_loader, desc="Predicting"):
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)

    # --- Aggregate logits by trial code ---
    trial_codes = dataset.labels.numpy()
    trial_logits = {}
    for idx, code in enumerate(trial_codes):
        code = int(code)  # Ensure code is a standard integer
        if code not in trial_logits:
            trial_logits[code] = []
        trial_logits[code].append(all_logits[idx])

    # --- Confidence-Weighted Aggregation and Prediction ---
    results = []
    for code in sorted(trial_logits.keys()):
        # Stack all logits for the current trial into a single tensor
        logits_tensor = torch.from_numpy(np.array(trial_logits[code]))[2:3, :]

        # If a trial has no valid windows, skip it (edge case)
        if logits_tensor.shape[0] == 0:
            continue

        # Convert logits to probabilities using softmax
        probs = F.softmax(logits_tensor, dim=1)

        # The confidence is the max probability for each window
        # We raise it to an exponent to give more weight to higher confidences
        confidences = torch.max(probs, dim=1).values ** confidence_exponent

        # If all confidences are zero, fall back to a simple mean to avoid division by zero
        if torch.sum(confidences) == 0:
            avg_logits = torch.mean(logits_tensor, dim=0).numpy()
        else:
            # Calculate the weighted average of the logits
            weighted_sum_of_logits = torch.sum(logits_tensor * confidences.unsqueeze(1), dim=0)
            sum_of_weights = torch.sum(confidences)
            avg_logits = (weighted_sum_of_logits / sum_of_weights).numpy()

        # Get the final prediction
        pred_idx = np.argmax(avg_logits)
        pred_label = decode_label(pred_idx)

        subj, sess, trial = position_decode(code)
        csv_id = id_lookup.get((subj, sess, trial))

        if csv_id is not None:
            results.append({"id": csv_id, "label": pred_label})

    # --- Save to CSV ---
    df = pd.DataFrame(results)
    df = df.sort_values("id")
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()
