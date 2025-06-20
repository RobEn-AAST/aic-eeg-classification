import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from modules.utils import evaluate_model, split_and_get_loaders
import optuna
from modules import EEGDataset


class Trainer:
    def __init__(self, data_path, train_epochs=1000, tune_epochs=30, optuna_n_trials=120):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_epochs = train_epochs
        self.tune_epochs = tune_epochs
        self.optuna_n_trials = optuna_n_trials

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.trial = None
        self.model: nn.Module = None # type: ignore

        self.train_loader = None
        self.eval_loader = None
        self.test_loader = None
        self.dataset = None

        self.storage = "sqlite:///optuna_studies.db"
        self.study_name = "ssvep_classifier_optimization"
        self.data_path = data_path


        self.checkpoint_path = "./checkpoints/ssvep"
        os.makedirs(os.path.join(self.checkpoint_path, "models"), exist_ok=True)
        self.checkpoint_model_path = os.path.join(self.checkpoint_path, "models")

    def _train_loop(self, n_epochs: int, should_save=False, should_print=False):
        assert isinstance(self.optimizer, torch.optim.Optimizer), "optimizer is not a valid optimizer"
        assert isinstance(self.train_loader, DataLoader), "train_laoder is not valid Datloader"
        if self.trial is None:
            print("Warning: self.trial is none, we are probably in acutal training phase")

        for epoch in range(n_epochs):
            self.model.to(self.device)
            self.model.train()

            avg_loss = 0
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)  # B x out_size
                loss = self.criterion(y_pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

            avg_loss = avg_loss / len(self.train_loader)
            evaluation = evaluate_model(self.model, self.val_loader, self.device)

            if self.trial is not None:
                self.trial.report(evaluation, epoch)
                if self.trial.should_prune():
                    optuna.exceptions.TrialPruned()

            if should_print:
                print(f"epoch {epoch}, evaluation {evaluation}, avg_loss {avg_loss}")

            if should_save:
                self.model.cpu()
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_model_path, f"ssvep.pth"))
                self.model.to(self.device)

    def _prepare_training(self, is_trial, stride_factor=2):
        if is_trial:
            assert isinstance(self.trial, optuna.Trial), "trial is none, cant' suggest params"

            window_length = self.trial.suggest_categorical("window_length", [175, 250, 350])
            batch_size = self.trial.suggest_categorical("batch_size", [32, 64])

        else:
            best_params = self._get_study().best_params
            window_length = best_params['window_length']
            batch_size = best_params["batch_size"]

        stride = int(window_length // stride_factor)
        self.dataset = EEGDataset(data_path=self.data_path, window_length=window_length, stride=stride)

        self.train_loader, self.val_loader, self.test_loader = split_and_get_loaders(self.dataset, batch_size)

    def _objective(self, trial: optuna.Trial):
        self.trial = trial
        self._prepare_training(True)
        assert self.model is not None, "model is not set, can't train, ensure to set it after overriding"
        assert self.optimizer is not None, "optimizer is not set, can't train, ensure to set it after overriding"

        self._train_loop(self.tune_epochs, should_save=False, should_print=False)
        evaluation = evaluate_model(self.model, self.val_loader, self.device)
        return evaluation

    def _get_study(self):
        return optuna.create_study(study_name=self.study_name, storage=self.storage, direction="maximize", load_if_exists=True)

    def optimize(self, delete_existing=False):
        if delete_existing:
            try:
                optuna.delete_study(study_name=self.study_name, storage=self.storage)
            except Exception:
                pass

        study = self._get_study()
        study.optimize(self._objective, n_trials=self.optuna_n_trials)

        # Print optimization results
        print("\nStudy statistics:")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
        print(f"  Number of complete trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")

        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("\nBest hyperparameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")

        return study.best_params

    def train(self):
        self.trial = None
        self._prepare_training(False)

        self._train_loop(self.train_epochs, should_save=True, should_print=True)
        evaluation = evaluate_model(self.model, self.val_loader, self.device)
        print("done training")
        return evaluation

if __name__ == '__main__':
    trainer = Trainer('./data/mtcaic3')
    print('hi')