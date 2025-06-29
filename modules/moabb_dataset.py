# %%
# for data info http://moabb.neurotechx.com/docs/dataset_summary.html
import moabb
from moabb.datasets import BNCI2014_001  # 250 hz
from moabb.datasets import BNCI2014_004  # 250 hz
from moabb.datasets import Zhou2016  # 250 hz
from moabb.paradigms import LeftRightImagery
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from mne import create_info
from mne.io import RawArray
from moabb.datasets.base import BaseDataset
import mne
import os


# %%
data_path = "/home/zeyadcode/Workspace/ai_projects/eeg_detection/data/mtcaic3"
split = "train"


class CompetitionDataset(BaseDataset):
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 30)),  # List of subject IDs
            sessions_per_subject=1,  # Number of sessions per subject
            events={"left_hand": 1, "right_hand": 2},
            code="CompetitionDataset",
            interval=[0, 4],  # Time interval for trials
            paradigm="imagery",  # "ssvep" or "imagery" or "p300"
            doi=None,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject - THIS IS THE KEY METHOD"""
        if self.paradigm == "ssvep":
            ch_names = ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]
            moabb_channels = ["FZ", "C3", "Cz", "C4", "PZ", "PO7", "OZ", "PO8"]
            task = "SSVEP"
        elif self.paradigm == "imagery":
            task = "MI"
            ch_names = ["C3", "C4", "CZ"]
            moabb_channels = ["C3", "C4", "Cz"]
        else:
            raise ValueError(f"got unexpected paradigm {self.paradigm}")

        # Load labels for this subject
        labels_df = pd.read_csv(os.path.join(data_path, f"{split}.csv"), usecols=["subject_id", "trial_session", "trial", "task", "label"])
        task_df = labels_df.query(f"task=='{task}' and subject_id=='S{subject}'")

        if task_df.empty:
            return {"0": {"0": None}}  # No data for this subject

        sfreq = 250
        ch_types = ["eeg"] * len(ch_names)

        # Process each session
        sessions = {}
        for session_id in task_df["trial_session"].unique():
            session_trials = task_df[task_df["trial_session"] == session_id]

            # Load EEG data for this session
            fp = os.path.join(data_path, task, split, f"S{subject}", str(session_id), "EEGdata.csv")
            if not os.path.exists(fp):
                continue

            # Load the full session data
            eeg_data = pd.read_csv(fp, usecols=ch_names).values
            total_samples = eeg_data.shape[0]
            trial_length = total_samples // 10  # 10 trials per session

            # Create continuous data and events
            all_trial_data = []
            events_list = []
            current_sample = 0

            for _, trial_row in session_trials.iterrows():
                trial_num = int(trial_row["trial"])

                # Extract trial data (trial numbers are 1-indexed)
                start_idx = (trial_num - 1) * trial_length
                end_idx = trial_num * trial_length
                trial_data = eeg_data[start_idx:end_idx]

                all_trial_data.append(trial_data)

                # Create event at trial start
                label = "left_hand" if trial_row.label == "Left" else "right_hand"
                events_list.append([current_sample, 0, self.event_id[label]])
                current_sample += trial_data.shape[0]

            if not all_trial_data:
                continue

            # Concatenate all trials for this session
            continuous_data = np.vstack(all_trial_data).T  # Shape: (channels, samples)

            # Create MNE info object
            info = create_info(moabb_channels, sfreq, ch_types)

            # Create Raw object (convert to microvolts)
            raw = RawArray(continuous_data * 1e-6, info, verbose=False)

            # Add events as annotations
            if events_list:
                events_array = np.array(events_list)
                event_desc = {v: k for k, v in self.event_id.items()}
                annotations = mne.annotations_from_events(events_array, sfreq=sfreq, event_desc=event_desc)
                raw.set_annotations(annotations)

            sessions[str(session_id)] = {"0": raw}

        if sessions is None:
            raise ValueError(f"Sessions is None for subject {subject}")
        return sessions if sessions else {"0": {"0": None}}
        # Return in required format: {"session_id": {"run_id": raw}}  

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """Return file paths for the subject's data"""
        subject_paths = []

        # Get all session directories for this subject
        task = "mi" if self.paradigm == "imagery" else "ssvep"
        subject_dir = os.path.join(data_path, task, split, f"S{subject}")
        if os.path.exists(subject_dir):
            for session in os.listdir(subject_dir):
                session_path = os.path.join(subject_dir, session)
                eeg_file = os.path.join(session_path, "EEGdata.csv")
                if os.path.exists(eeg_file):
                    subject_paths.append(eeg_file)

        return subject_paths


# %%
paradigm = LeftRightImagery(
    channels=["Cz", "C3", "C4"],
    tmin=0.0,
    tmax=4.0,
    resample=250,
)

# %%
X, labels, metadata = paradigm.get_data(CompetitionDataset())

combined_labels = []
for i in range(len(labels)):
    if labels[i]  == 'left_hand':
        class_label = 0
    elif labels[i]  == 'right_hand':
        class_label = 1
    else:
        raise ValueError(f"Unexpected label {labels[i]}")

    subject_num = metadata.iloc[i]["subject"]  # Subject number from metadata
    combined_labels.append([class_label, subject_num])

# %%
if __name__ == '__main__':
    dataset1 = BNCI2014_001()
    dataset2 = BNCI2014_004()
    dataset3 = Zhou2016()

    print(f"channels: {dataset1.get_data()[1]['0train']['0'].info.ch_names}")
    print(f"classes: {BNCI2014_001().event_id}")
    print(f"classes: {BNCI2014_004().event_id}")
    print(f"classes: {Zhou2016().event_id}")
