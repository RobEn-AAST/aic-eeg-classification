import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import optuna
from optuna.distributions import IntUniformDistribution, FloatDistribution, CategoricalDistribution


def evaluate_model(model: nn.Module, data_loader: DataLoader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return float(accuracy)


def split_and_get_loaders(dataset: Dataset, batch_size: int):
    """
    gets dataset and splits it into train, validation, test sets
    returns train dataloader, validation dataloader, test dataloader
    """
    X = dataset.data.numpy()  # type: ignore
    Y = dataset.labels.numpy()  # type: ignore

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y,
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val,
        Y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=Y_train_val,
    )

    # Turn to tensors
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).long()

    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).long()

    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).long()

    # Build dataset
    train_ds = TensorDataset(X_train_t, Y_train_t)
    val_ds = TensorDataset(X_val_t, Y_val_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader, test_loader


def manual_write_study_params(study_name: str, storage: str):
    params = {
        "window_length": 128,   # or 160
        "stride": 2,            # between 2-3
        "hidden_size": 128,     # 64-192 step 32
        "num_layers": 2,        # 1-3
        "dropout": 0.2,         # 0.0-0.4
        "lr": 0.001,            # 3e-4 to 3e-2
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