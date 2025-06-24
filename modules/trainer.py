import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split
from modules.utils import evaluate_model # Assuming this function exists and is correct
import optuna
from modules import EEGDataset # Assuming this class exists and is correct
import numpy as np

class Trainer:
    def __init__(self, data_path, optuna_db_path, model_path, train_epochs=1000, tune_epochs=30, optuna_n_trials=120, data_fraction=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_epochs = train_epochs
        self.tune_epochs = tune_epochs
        self.optuna_n_trials = optuna_n_trials

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.trial = None
        self.model: nn.Module = None
        self.scheduler = None

        # <<< FIX: Renamed eval_loader to val_loader for consistency
        self.train_loader = None
        self.val_loader = None 
        self.dataset = None
        self.data_fraction = data_fraction

        self.storage = f"sqlite:///{optuna_db_path}"
        self.study_name = "ssvep_classifier_optimization"
        self.data_path = data_path
        self.model_path = model_path

    def _train_loop(self, n_epochs: int, should_save=False, should_print=False):
        assert isinstance(self.optimizer, torch.optim.Optimizer), "Optimizer is not valid"
        assert isinstance(self.train_loader, DataLoader), "train_loader is not a valid DataLoader"
        assert isinstance(self.val_loader, DataLoader), "val_loader is not a valid DataLoader"

        # This warning is helpful, so we keep it.
        if self.trial is None:
            print("Warning: self.trial is None. Assuming this is the final training phase.")

        for epoch in range(n_epochs):
            self.model.to(self.device)
            self.model.train()

            avg_loss = 0
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
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
                    # <<< FIX: You must raise the exception for it to work.
                    raise optuna.exceptions.TrialPruned()

            if self.scheduler is not None:
                self.scheduler.step(evaluation)

            if should_print:
                print(f"Epoch {epoch}/{n_epochs}, Validation Accuracy: {evaluation:.4f}, Avg Loss: {avg_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']}")

            # <<< FIX: Moved saving outside the loop. You should only save the final model after all epochs.
            if should_save and epoch % 5 == 0:
                self.model.cpu()
                torch.save(self.model.state_dict(), self.model_path)
                self.model.to(self.device)
                print(f"Model saved to {self.model_path}")


    # <<< FIX: This entire method was refactored for clarity and correctness.
    def _prepare_data(self, is_trial, batch_size=None, window_length=None, stride_factor=3):
        # This logic is now handled by the CustomTrainer override.
        # This base method just handles creating datasets and dataloaders.
        if is_trial:
            assert isinstance(self.trial, optuna.Trial), "self.trial must be an Optuna.Trial during optimization."
            # Suggest parameters if they are not provided (they will be provided by the CustomTrainer)
            window_length = window_length or self.trial.suggest_categorical("window_length", [128, 256, 640])
            batch_size = batch_size or self.trial.suggest_categorical("batch_size", [32, 64])
        else:
            # For the final training, get the best parameters from the study.
            study = self._get_study()
            best_params = study.best_params
            window_length = window_length or best_params["window_length"]
            batch_size = batch_size or best_params["batch_size"]

        stride = int(window_length // stride_factor)

        # <<< FIX: Simplified data loading. Load all relevant data once.
        # Assuming EEGDataset can handle loading all data without a 'split' argument
        # or that you handle train/val splits externally. The Concat+random_split is a good pattern.
        
        # This assumed 'split' argument might need to be adjusted based on your EEGDataset implementation

        dataset_train_full = EEGDataset(
            self.data_path,
            window_length=window_length,
            stride=stride,
            data_fraction=self.data_fraction,
            hardcoded_mean=True,
        )

        dataset_val_full = EEGDataset(
            data_path=self.data_path,
            window_length=window_length,
            stride=stride,
            task='ssvep',
            split='validation',
            read_labels=True,
            hardcoded_mean=True,
            data_fraction=1
        )
        # <<< FIX: Assign the concatenated dataset to self.dataset so CustomTrainer can use it.
        self.dataset = ConcatDataset([dataset_train_full, dataset_val_full])
        
        train_len = int(len(self.dataset) * 0.8)
        val_len = len(self.dataset) - train_len
        
        train_ds, val_ds = random_split(self.dataset, [train_len, val_len])
        
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        print(f"Data prepared: Train batches={len(self.train_loader)}, Val batches={len(self.val_loader)}")
        


    def _objective(self, trial: optuna.Trial):
        self.trial = trial
        
        # In your CustomTrainer, you will override a method that calls _prepare_data
        # and then sets the model and optimizer.
        # For this to work, we must call the custom preparation logic.
        self.prepare_trial_run() # This method will be defined in CustomTrainer
        
        assert self.model is not None, "Model is not set. Ensure prepare_trial_run creates self.model."
        assert self.optimizer is not None, "Optimizer is not set. Ensure prepare_trial_run creates self.optimizer."
 
        self._train_loop(self.tune_epochs, should_print=False)
        evaluation = evaluate_model(self.model, self.val_loader, self.device)
        return evaluation

    def _get_study(self):
        return optuna.create_study(study_name=self.study_name, storage=self.storage, direction="maximize", load_if_exists=True)

    def optimize(self, delete_existing=False, should_print=False):
        if delete_existing:
            try:
                optuna.delete_study(study_name=self.study_name, storage=self.storage)
                print(f"Study '{self.study_name}' deleted.")
            except Exception:
                print("Could not delete study (it might not exist).")

        study = self._get_study()
        study.optimize(self._objective, n_trials=self.optuna_n_trials)

        print("\n--- Optimization Finished ---")
        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print(f"  Number of pruned trials: {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Best hyperparameters: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study.best_params

    def train(self):
        self.trial = None
        # Same as in _objective, we rely on the custom implementation
        self.prepare_final_run()

        self._train_loop(self.train_epochs, should_save=True, should_print=True)
        evaluation = evaluate_model(self.model, self.val_loader, self.device)
        print(f"--- Final Training Done ---")
        print(f"Final evaluation accuracy: {evaluation}")
        return evaluation