import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split
from modules.utils import evaluate_model  # Assuming this function exists and is correct
import optuna
from modules import EEGDataset  # Assuming this class exists and is correct
import numpy as np
from torch.utils.data import Subset, DataLoader


class Trainer:
    def __init__(self, data_path, optuna_db_path, model_path, task, eeg_channels, train_epochs=1000, tune_epochs=30, optuna_n_trials=120, data_fraction=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_epochs = train_epochs
        self.tune_epochs = tune_epochs
        self.optuna_n_trials = optuna_n_trials
        self.task = task
        self.eeg_channels = eeg_channels

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.trial = None
        self.model: nn.Module = None
        self.scheduler = None

        # <<< FIX: Renamed eval_loader to val_loader for consistency
        self.train_loader = None
        self.val_loader = None
        self.data_fraction = data_fraction

        self.storage = f"sqlite:///{optuna_db_path}"
        self.study_name = f"{task}_classifier_optimization"
        self.data_path = data_path
        self.model_path = model_path

    def _train_loop(self, n_epochs: int, should_save=False, should_print=False):
        assert isinstance(self.optimizer, torch.optim.Optimizer), "Optimizer is not valid"
        assert isinstance(self.train_loader, DataLoader), "train_loader is not a valid DataLoader"
        assert isinstance(self.val_loader, DataLoader), "val_loader is not a valid DataLoader"
        assert isinstance(self.target_loader, DataLoader), "target_loader is not a valid DataLoader"

        # If you created separate optimizers in prepare_trial_run:
        opt_FE, opt_F, opt_Fp = self.opt_FE, self.opt_F, self.opt_Fp
        criterion = self.criterion

        # Warn if final run
        if self.trial is None:
            print("Warning: self.trial is None. Assuming this is the final training phase.")

        for epoch in range(n_epochs):
            self.model.to(self.device)
            self.model.train()

            avg_loss_label  = 0.0
            avg_loss_domain = 0.0
            correct_label   = 0
            total           = 0

            # ——— A) Source classification ———
            for x_s, y_s in self.train_loader:
                x_s      = x_s.to(self.device)
                y_labels = y_s[:, 0].to(self.device)

                out1, _ = self.model(x_s)
                loss_cls = criterion(out1, y_labels)

                opt_FE.zero_grad(); opt_F.zero_grad()
                loss_cls.backward()
                opt_FE.step();  opt_F.step()

                avg_loss_label += loss_cls.item()
                _, pred = out1.max(1)
                correct_label  += (pred == y_labels).sum().item()
                total          += y_labels.size(0)

            # ——— B) Adversarial maximization for F' ———
            for x_t, _ in self.target_loader:
                x_t = x_t.to(self.device)

                with torch.no_grad():
                    logits_t, _ = self.model(x_t)
                yhat = logits_t.argmax(dim=1)

                # forward through F'
                seq = self.model.feature_extractor(x_t)
                out2 = self.model.Fp(seq)
                top2 = out2.topk(2, dim=1).values
                margin = top2[:, 1] - top2[:, 0]
                loss_max = -(margin.mean())

                opt_Fp.zero_grad()
                loss_max.backward()
                opt_Fp.step()

            # ——— C) Alignment minimization for FE + F ———
            for x_t, _ in self.target_loader:
                x_t = x_t.to(self.device)

                _, out2_t = self.model(x_t)
                top2 = out2_t.topk(2, dim=1).values
                margin = top2[:, 1] - top2[:, 0]
                loss_mdd = margin.mean()

                avg_loss_domain += loss_mdd.item()

                opt_FE.zero_grad(); opt_F.zero_grad()
                loss_mdd.backward()
                opt_FE.step();  opt_F.step()

            # ——— Compute metrics & optional saving ———
            avg_loss_label  /= len(self.train_loader)
            avg_loss_domain /= len(self.target_loader)
            train_acc = 100.0 * correct_label / total

            # validation
            val_acc, _ = evaluate_model(self.model, self.val_loader, self.device)

            # report/prune
            if self.trial is not None:
                self.trial.report(val_acc, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step(val_acc)

            if should_print:
                lr = opt_FE.param_groups[0]['lr']
                print(f"Epoch {epoch}/{n_epochs}, Val Acc: {val_acc:.4f}, "
                    f"Train Acc: {train_acc:.2f}%, "
                    f"AvgLabLoss: {avg_loss_label:.4f}, AvgDomLoss: {avg_loss_domain:.4f}, lr: {lr}")

            if should_save and (epoch + 1) % 5 == 0:
                self.model.cpu()
                torch.save(self.model.state_dict(), self.model_path)
                self.model.to(self.device)
                print(f"Model saved to {self.model_path}")
                
                
    def _prepare_data(self, is_trial, batch_size=None, window_length=None, stride_factor=3):
        if is_trial:
            assert isinstance(self.trial, optuna.Trial), "self.trial must be an Optuna.Trial during optimization."
            window_length = window_length or self.trial.suggest_categorical("window_length", [128, 256, 640])
            batch_size = batch_size or self.trial.suggest_categorical("batch_size", [32, 64])
        else:
            study = self._get_study()
            best_params = study.best_params
            window_length = window_length or best_params["window_length"]
            batch_size = batch_size or best_params["batch_size"]

        stride = int(window_length // stride_factor)
        
        dataset_train = EEGDataset(
            self.data_path, window_length=window_length, stride=stride, data_fraction=self.data_fraction, hardcoded_mean=True, task=self.task, eeg_channels=self.eeg_channels
        )

        dataset_val = EEGDataset(
            data_path=self.data_path, window_length=window_length, stride=stride, task=self.task, split="validation", read_labels=True, hardcoded_mean=True, eeg_channels=self.eeg_channels
        )
        
        target_subject = 29

        is_target = (dataset_train.subjects == target_subject)
        target_indices = torch.nonzero(is_target, as_tuple=True)[0].tolist()
        source_indices = torch.nonzero(~is_target, as_tuple=True)[0].tolist()

        self.train_loader = DataLoader(
            Subset(dataset_train, source_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        self.target_loader = DataLoader(
            Subset(dataset_train, target_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        
        self.val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        print(f"Data prepared: Train batches={len(self.train_loader)}, Val batches={len(self.val_loader)}")

    def _objective(self, trial: optuna.Trial):
        self.trial = trial

        self.prepare_trial_run()  # This method will be defined in CustomTrainer

        assert self.model is not None, "Model is not set. Ensure prepare_trial_run creates self.model."
        assert self.optimizer is not None, "Optimizer is not set. Ensure prepare_trial_run creates self.optimizer."

        self._train_loop(self.tune_epochs, should_print=True)
        evaluation_label, _ = evaluate_model(self.model, self.val_loader, self.device)
        return evaluation_label

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

        self.data_fraction = 1
        self._train_loop(self.train_epochs, should_save=True, should_print=True)
        evaluation = evaluate_model(self.model, self.val_loader, self.device)
        print(f"--- Final Training Done ---")
        print(f"Final evaluation accuracy: {evaluation}")
        return evaluation
