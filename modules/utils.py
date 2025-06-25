import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import optuna
from optuna.distributions import IntUniformDistribution, FloatDistribution, CategoricalDistribution


def get_closest_divisor(target: int, n=1750):
    n = n or 1750

    divisors = [i for i in range(1, n+1) if n % i == 0]
    closest = min(divisors, key=lambda x: abs(x - target))
    return closest


def evaluate_model(model: nn.Module, data_loader: DataLoader, device):
    """
    Evaluates both label and subject prediction accuracy.
    Assumes:
      - Dataset __getitem__ returns (x, y) where y is a tensor of shape (B, 2):
          y[:, 0] = true labels
          y[:, 1] = true subject IDs
      - model(x) returns a tuple: (label_logits, subject_logits)
    """
    model.eval()
    model.to(device)

    all_label_preds = []
    all_label_trues = []
    all_subj_preds = []
    all_subj_trues = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            # split ground truth
            y_label = y[:, 0]
            y_subject = y[:, 1]

            # forward pass: two heads
            label_logits, subject_logits = model(x)

            # predictions
            _, label_pred = torch.max(label_logits, dim=1)
            _, subject_pred = torch.max(subject_logits, dim=1)

            # collect on CPU
            all_label_preds.extend(label_pred.cpu().numpy())
            all_label_trues.extend(y_label.cpu().numpy())
            all_subj_preds.extend(subject_pred.cpu().numpy())
            all_subj_trues.extend(y_subject.cpu().numpy())

    # compute accuracies
    label_accuracy = accuracy_score(all_label_trues, all_label_preds)
    subject_accuracy = accuracy_score(all_subj_trues, all_subj_preds)

    return float(label_accuracy), float(subject_accuracy)



def split_and_get_loaders(dataset: Dataset, batch_size: int, train_size: float = 0.8):
    """
    gets dataset and splits it into train, validation, test sets
    returns train dataloader, validation dataloader, test dataloader
    """
    X = dataset.data.numpy()  # type: ignore
    Y = dataset.labels.numpy()  # type: ignore

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y,
    )

    X_train, X, Y_train, Y = train_test_split(
        X_train,
        Y_train,
        test_size=(1 - train_size),
        random_state=42,
        stratify=Y_train,
    )

    # Turn to tensors
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).long()

    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).long()

    # Build dataset
    train_ds = TensorDataset(X_train_t, Y_train_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def manual_write_study_params(study_name: str, storage: str):
    params = {
        "window_length": 128,   # or 160
        "stride": 2,            # between 2-3
        "hidden_size": 128,     # 64-192 step 32
        "num_layers": 2,        # 1-3
        "dropout": 0.2,         # 0.0-0.4
        "lr": 0.01,            # 3e-4 to 3e-2
        "batch_size": 32,       # 32 or 64
    }

    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
    )

    dists = {
        "window_length": IntUniformDistribution(128, 160),
        "stride": FloatDistribution(2, 3),
        "hidden_size": IntUniformDistribution(64, 192, step=32),
        "num_layers": IntUniformDistribution(1, 3),
        "dropout": FloatDistribution(0.0, 0.4),
        "lr": FloatDistribution(3e-4, 3e-2),
        "batch_size": CategoricalDistribution([32, 64]),
    }

    trial = optuna.create_trial(
        params=params,
        distributions=dists,
        value=0.0,
        state=optuna.trial.TrialState.COMPLETE,
    )

    study.add_trial(trial)

    return study.best_params


if __name__ == '__main__':
    print(get_closest_divisor(160))
    print(get_closest_divisor(200))
    print(get_closest_divisor(300))
    print(get_closest_divisor(400))
    print(get_closest_divisor(700))